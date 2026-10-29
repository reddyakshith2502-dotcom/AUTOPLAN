import numpy as np

class IntelligentDriverModel:
    """
    Intelligent Driver Model (IDM) for Longitudinal Car-Following Kinematics.
    Calibrated with empirical parameters from NGSIM US-101 highway driving data.
    
    Formula:
        dv/dt = a_max * [ 1 - (v / v0)^delta - (s* / s)^2 ]
        s*    = s0 + max(0, v * T_headway + (v * delta_v) / (2 * sqrt(a_max * b_comf)))
    """
    def __init__(
        self,
        desired_speed=15.0,    # v0: desired free-flow speed (m/s ~ 54 km/h)
        time_headway=1.2,      # T: safe time headway (seconds)
        min_spacing=3.0,       # s0: minimum jam distance at standstill (meters)
        max_accel=2.0,         # a_max: maximum comfortable acceleration (m/s^2)
        comfortable_decel=2.5, # b_comf: comfortable deceleration rate (m/s^2)
        accel_exponent=4.0     # delta: acceleration exponent
    ):
        self.v0 = desired_speed
        self.T = time_headway
        self.s0 = min_spacing
        self.a_max = max_accel
        self.b_comf = comfortable_decel
        self.delta = accel_exponent

    def compute_acceleration(self, ego_speed, leading_speed=None, distance_gap=None):
        """
        Computes the target instantaneous acceleration (m/s^2) for the ego vehicle.
        If no leading vehicle is present within sensory range, free-road acceleration is applied.
        """
        v = max(0.0, float(ego_speed))
        
        # Free road term
        free_accel = self.a_max * (1.0 - (v / max(0.1, self.v0)) ** self.delta)
        
        if distance_gap is None or leading_speed is None or distance_gap > 120.0:
            return float(np.clip(free_accel, -6.0, self.a_max))
        
        # Interaction term with leading vehicle
        s = max(0.5, float(distance_gap))
        delta_v = v - float(leading_speed)  # Closing speed (>0 if ego is closing in)
        
        # Desired dynamic distance s*
        s_star = self.s0 + max(
            0.0,
            (v * self.T) + (v * delta_v) / (2.0 * np.sqrt(self.a_max * self.b_comf) + 1e-6)
        )
        
        interaction_term = (s_star / s) ** 2
        accel = self.a_max * (1.0 - (v / max(0.1, self.v0)) ** self.delta - interaction_term)
        
        # Physical braking limits (emergency braking clamped to -6.5 m/s^2)
        return float(np.clip(accel, -6.5, self.a_max))


class QuinticSplinePlanner:
    """
    Quintic (5th-Order) Polynomial Spline Trajectory Generator.
    Ensures C^2 continuity (continuous position, continuous velocity, continuous curvature/acceleration)
    to eliminate abrupt corner snapping and unrealistic jerks at road turns.
    """
    @staticmethod
    def generate_quintic_curve(p_start, p_end, heading_start, heading_end, num_points=35):
        """
        Generates smooth curve points between p_start (x0, y0) and p_end (x1, y1)
        satisfying initial and final heading angles.
        """
        x0, y0 = float(p_start[0]), float(p_start[1])
        x1, y1 = float(p_end[0]), float(p_end[1])
        dist = np.hypot(x1 - x0, y1 - y0)
        
        if dist < 1.0:
            return [(x0, y0), (x1, y1)]
        
        scale = dist * 0.45
        dx0, dy0 = np.cos(heading_start) * scale, np.sin(heading_start) * scale
        dx1, dy1 = np.cos(heading_end) * scale, np.sin(heading_end) * scale
        
        # Hermite-like cubic/quintic interpolation
        points = []
        for t in np.linspace(0.0, 1.0, num_points):
            # Quintic blend weights
            h0 = 1.0 - 10.0 * (t**3) + 15.0 * (t**4) - 6.0 * (t**5)
            h1 = 10.0 * (t**3) - 15.0 * (t**4) + 6.0 * (t**5)
            h2 = t - 6.0 * (t**3) + 8.0 * (t**4) - 3.0 * (t**5)
            h3 = -4.0 * (t**3) + 7.0 * (t**4) - 3.0 * (t**5)
            
            px = h0 * x0 + h1 * x1 + h2 * dx0 + h3 * dx1
            py = h0 * y0 + h1 * y1 + h2 * dy0 + h3 * dy1
            points.append((float(px), float(py)))
            
        return points


class KinematicBicycleModel:
    """
    Kinematic Bicycle Model for Autonomous Vehicle Motion.
    State: [x, y, yaw, velocity]
    Control: [acceleration, steering_angle]
    """
    def __init__(self, wheelbase=2.8, max_steer_rad=0.6):
        self.L = wheelbase
        self.max_steer = max_steer_rad

    def step(self, x, y, yaw, v, accel, steer_angle, dt=0.05):
        steer = np.clip(steer_angle, -self.max_steer, self.max_steer)
        
        # Update position & heading
        x_new = x + v * np.cos(yaw) * dt
        y_new = y + v * np.sin(yaw) * dt
        yaw_new = yaw + (v / self.L) * np.tan(steer) * dt
        
        # Normalize yaw to [-pi, pi]
        yaw_new = np.arctan2(np.sin(yaw_new), np.cos(yaw_new))
        
        # Update speed
        v_new = max(0.0, v + accel * dt)
        
        return x_new, y_new, yaw_new, v_new
