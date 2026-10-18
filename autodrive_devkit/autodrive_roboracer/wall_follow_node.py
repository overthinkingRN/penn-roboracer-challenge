#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
import math

class VelocityBoundedArcController(Node):
    def __init__(self):
        super().__init__('velocity_bounded_arc_node')
        
        self.throttle_pub = self.create_publisher(Float32, '/autodrive/throttle', 10)
        self.steering_pub = self.create_publisher(Float32, '/autodrive/steering', 10)
        
        self.lidar_sub = self.create_subscription(
            LaserScan, 
            '/autodrive/lidar', 
            self.lidar_callback, 
            10
        )
        
        # --- CALIBRATED RACING PROFILE ---
        self.DESIRED_LEFT_DIST = 0.45  # Target distance from left track border
        
        # Speed Bounds
        self.MAX_THROTTLE = 0.18       # Max straight speed to prevent kinetic plowing
        self.CORNER_THROTTLE = 0.03    # Controlled corner entry crawling velocity
        self.BRAKE_FORCE = -0.15       # Active reverse engine braking to plant front tires
        
        # Steering Caps
        self.MAX_STEER_ANGLE = 0.68    # Sufficient throw for hairpins without tire scrubbing
        self.KP = 2.4                  # Stable baseline proportional tracking factor

        self.get_logger().info("Velocity-Bounded Arc Controller Initialized.")

    def lidar_callback(self, scan_msg):
        # 1. Map explicit spatial points (0° is forward, Positive is Left)
        forward_dist  = self.get_distance_at_angle(scan_msg, angle_deg=0)
        parallel_left = self.get_distance_at_angle(scan_msg, angle_deg=90)
        
        # 2. Comprehensive Fan Scan (Sweeping the 0° to 60° quadrant)
        # We look for the absolute closest constraint blocking the car's potential path.
        fan_angles = [0, 15, 30, 45, 60]
        closest_wall_in_fan = 3.0
        
        for angle in fan_angles:
            dist = self.get_distance_at_angle(scan_msg, angle_deg=angle)
            if dist < closest_wall_in_fan:
                closest_wall_in_fan = dist

        # 3. Proactive Velocity Mapping (Prevents Outer Wall Plowing)
        if closest_wall_in_fan < 0.9 or forward_dist < 1.1:
            # IMMEDIATE RISK ZONE: Execute engine brake to shift mass to front steering axle
            target_throttle = self.BRAKE_FORCE
        elif closest_wall_in_fan < 1.6 or forward_dist < 1.8:
            # APPROACH PHASE: Drop throttle down completely to settle chassis
            target_throttle = self.CORNER_THROTTLE
        else:
            # STRAIGHTAWAY MODE: Accelerate smoothly
            target_throttle = self.MAX_THROTTLE

        # 4. Phase-Corrected Error Synthesis
        # We blend the current lateral error with a projection of the 45° upcoming apex layout
        look_ahead_45 = self.get_distance_at_angle(scan_msg, angle_deg=45)
        
        base_error = parallel_left - self.DESIRED_LEFT_DIST
        predictive_bias = 0.0
        
        if look_ahead_45 < 1.3:
            # Push the car into the turn early before arriving at the corner entry
            predictive_bias += (1.3 - look_ahead_45) * 0.75
            
        total_error = base_error + predictive_bias

        # 5. Anti-Overcorrection Constraint (Prevents Inner Wall Clipping)
        # If the vehicle's left flank is getting closer than 0.35m to the inner boundary,
        # we forcefully depress the steering gain to let the front tires track wide.
        if parallel_left < 0.35:
            # Override predictive bias; focus strictly on local avoidance pushing the vehicle away
            total_error = parallel_left - self.DESIRED_LEFT_DIST
            # Multiply response to push the steering wheels straight/right immediately
            steering_angle = total_error * 3.5 
        else:
            # Standard tracking control law execution
            steering_angle = total_error * self.KP

        # 6. Smooth Actuation Bounds Clipping
        steering_angle = max(min(steering_angle, self.MAX_STEER_ANGLE), -self.MAX_STEER_ANGLE)

        # 7. Issue Actuation Real-time
        self.publish_commands(target_throttle, steering_angle)

    def get_distance_at_angle(self, scan, angle_deg):
        angle_rad = math.radians(angle_deg)
        if scan.angle_increment == 0:
            return 3.0
        idx = int((angle_rad - scan.angle_min) / scan.angle_increment)
        
        if 0 <= idx < len(scan.ranges):
            val = scan.ranges[idx]
            if math.isnan(val) or math.isinf(val) or val < scan.range_min:
                return scan.range_max
            return val
        return 3.0

    def publish_commands(self, throttle, steering):
        t_msg = Float32()
        t_msg.data = float(throttle)
        s_msg = Float32()
        s_msg.data = float(steering)
        self.throttle_pub.publish(t_msg)
        self.steering_pub.publish(s_msg)

    def stop_vehicle(self):
        self.publish_commands(0.0, 0.0)

def main(args=None):
    rclpy.init(args=args)
    node = VelocityBoundedArcController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_vehicle()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
