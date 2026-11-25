"""Gazebo factory + N TurtleBot3 robots (by robot type: Waffle + OpenMANIPULATOR-X arm, or a tray robot served by handlers) with namespaced Nav2 + the FactorySim node.

localization:=truth (default) uses ground-truth localisation (Gazebo odometry is world-sourced, so map -> odom is the
identity); localization:=amcl runs AMCL on the lidar like a real robot.

    ros2 launch factory_gazebo.launch.py instance:=lab robots:=3 orders:=8 policy:=edd gui:=false out:=/tmp/run
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, ExecuteProcess, GroupAction, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler, Shutdown, TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

ARGS = {"instance": "lab", "robots": "3", "orders": "0", "policy": "edd", "seed": "0", "proc_scale": "1.0", "gui": "false", "out": "/tmp/factory_sim_run", "delay": "30.0", "nominal": "false", "record": "false", "localization": "truth", "disturbances": "false", "charge_scale": "1.0", "teleop_mode": "auto", "people": "true"}
NAV2_TUNING = {"xy_goal_tolerance: 0.25": "xy_goal_tolerance: 0.15", "yaw_goal_tolerance: 0.25": "yaw_goal_tolerance: 0.2", "inflation_radius: 0.55": "inflation_radius: 0.45", "required_movement_radius: 0.5": "required_movement_radius: 0.25", "movement_time_allowance: 10.0": "movement_time_allowance: 20.0"}


def share(pkg):
    try:
        return get_package_share_directory(pkg)
    except Exception:
        return None


def setup(context):
    from factory_sim.ros import robot, world
    from factory_sim.ros.factory_node import scenario
    v = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    os.makedirs(v["out"], exist_ok=True)
    cfg, jobs, _ = scenario(v["instance"], int(v["robots"]), int(v["orders"]), int(v["seed"]))
    mesh_dir = next((d for pkg in ("turtlebot3_manipulation_gazebo", "turtlebot3_manipulation_description", "open_manipulator_x_description") if share(pkg)
                     for d in (os.path.join(share(pkg), "meshes", "open_manipulator_x"), os.path.join(share(pkg), "meshes")) if os.path.exists(os.path.join(d, "link1.stl"))), None)
    files = world.write(cfg, v["out"], jobs=jobs, record=v["record"] == "true", mesh_dir=mesh_dir, people=v["people"] == "true")
    nav2 = share("nav2_bringup")
    params = open(os.path.join(nav2, "params", "nav2_multirobot_params_all.yaml")).read()
    for a, b in NAV2_TUNING.items():
        params = params.replace(a, b)
    libs = [p for p in os.environ.get("LD_LIBRARY_PATH", "/opt/ros/humble/lib").split(":") if p]
    params = "\n".join(l for l in params.splitlines() if not l.strip().endswith("_bt_node") or any(os.path.exists(os.path.join(d, "lib{}.so".format(l.strip().lstrip("- ").strip()))) for d in libs))
    if v["localization"] == "truth":
        params = params.replace("tf_broadcast: true", "tf_broadcast: false")
    params += "\nmap_server:\n  ros__parameters:\n    use_sim_time: True\n    yaml_filename: {}\n".format(files["map"])
    params_file = os.path.join(v["out"], "nav2_params.yaml")
    open(params_file, "w").write(params)
    waffle = open(os.path.join(nav2, "worlds", "waffle.model")).read()
    acts = []
    tb3 = share("turtlebot3_gazebo")
    if tb3:
        acts.append(AppendEnvironmentVariable("GAZEBO_MODEL_PATH", os.path.join(tb3, "models")))
    acts.append(ExecuteProcess(cmd=["gzserver", "-s", "libgazebo_ros_init.so", "-s", "libgazebo_ros_factory.so", files["world"]], output="screen"))
    if v["gui"] == "true":
        acts.append(ExecuteProcess(cmd=["gzclient"], output="screen"))
    for i, (x, y, yaw) in enumerate(world.spawn(world.Factory(cfg))):
        name = "robot{}".format(i + 1)
        sdf_path = os.path.join(v["out"], name + ".sdf")
        open(sdf_path, "w").write(robot.robot_sdf(waffle, os.path.join(tb3, "models") if tb3 else None))
        if v["localization"] == "truth":
            acts.append(Node(package="tf2_ros", executable="static_transform_publisher", name="map_to_odom", namespace=name, arguments=["--frame-id", "map", "--child-frame-id", "odom"], remappings=[("/tf", "/{}/tf".format(name)), ("/tf_static", "/{}/tf_static".format(name))], parameters=[{"use_sim_time": True}]))
        acts.append(GroupAction([IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(nav2, "launch", "tb3_simulation_launch.py")), launch_arguments={
            "namespace": name, "use_namespace": "True", "map": files["map"], "use_sim_time": "True", "params_file": params_file, "autostart": "True", "use_rviz": "False", "use_simulator": "False",
            "headless": "True", "use_robot_state_pub": "True", "robot_sdf": sdf_path, "robot_name": name, "x_pose": str(x), "y_pose": str(y), "z_pose": "0.01", "roll": "0.0", "pitch": "0.0", "yaw": str(yaw)}.items())]))
    cmd = ["python3", "-m", "factory_sim.ros.factory_node", "--instance", v["instance"], "--robots", v["robots"], "--orders", v["orders"], "--policy", v["policy"], "--seed", v["seed"], "--proc-scale", v["proc_scale"],
           "--layout", files["layout"], "--sim-time", "--charge-scale", v["charge_scale"], "--teleop-mode", v["teleop_mode"], "--out", os.path.join(v["out"], "result.json")] + (["--nominal"] if v["nominal"] == "true" else []) + (["--disturbances"] if v["disturbances"] == "true" else [])
    factory = ExecuteProcess(cmd=cmd + ["--ros-args", "-p", "use_sim_time:=true"], output="screen")
    acts.append(TimerAction(period=float(v["delay"]), actions=[factory]))
    acts.append(RegisterEventHandler(OnProcessExit(target_action=factory, on_exit=[Shutdown(reason="factory finished")])))
    return acts


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument(k, default_value=d) for k, d in ARGS.items()] + [OpaqueFunction(function=setup)])
