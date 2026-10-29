import numpy as np

class VTMicroEnergyModel:
    """
    Virginia Tech Microscopic (VT-Micro) Energy & Fuel Consumption Model.
    Estimates instantaneous electric/fuel energy consumption rate (kWh/100km or L/h)
    as a function of instantaneous velocity (v) and longitudinal acceleration (a).
    """
    def __init__(self, vehicle_mass_kg=1600.0, drag_coeff=0.28, frontal_area_m2=2.2):
        self.mass = vehicle_mass_kg
        self.cd = drag_coeff
        self.area = frontal_area_m2
        self.air_density = 1.225
        self.cr = 0.015  # Rolling resistance
        self.g = 9.81
        self.powertrain_efficiency = 0.88
        self.regen_efficiency = 0.65

    def compute_power_kw(self, v_mps, a_mps2):
        """
        Computes tractive power at the wheels (kW).
        P_tractive = [m * a + m * g * Cr + 0.5 * rho * Cd * A * v^2] * v
        """
        v = max(0.0, float(v_mps))
        a = float(a_mps2)
        
        # Resistance forces
        f_inertial = self.mass * a
        f_rolling = self.mass * self.g * self.cr
        f_aero = 0.5 * self.air_density * self.cd * self.area * (v ** 2)
        
        f_total = f_inertial + f_rolling + f_aero
        p_wheels_kw = (f_total * v) / 1000.0  # Watts to kW
        
        if p_wheels_kw >= 0:
            p_battery_kw = p_wheels_kw / self.powertrain_efficiency
        else:
            # Regenerative braking recovery
            p_battery_kw = p_wheels_kw * self.regen_efficiency
            
        return float(p_battery_kw)

    def compute_energy_rate_kwh_per_100km(self, v_mps, a_mps2):
        """Calculates normalized energy consumption rate in kWh / 100 km"""
        v_kmh = max(0.5, float(v_mps) * 3.6)
        p_kw = self.compute_power_kw(v_mps, a_mps2)
        
        if p_kw <= 0:
            return float(np.clip(p_kw * (100.0 / v_kmh), -12.0, 0.0))
        
        rate = (p_kw / v_kmh) * 100.0
        return float(np.clip(rate, 2.0, 45.0))


class RideComfortEvaluator:
    """
    ISO 2631 Passenger Ride Comfort Evaluation Engine.
    Quantifies ride comfort using instantaneous Jerk (da/dt) and Root Mean Square (RMS) acceleration.
    """
    @staticmethod
    def classify_jerk_comfort(jerk_mps3):
        j = abs(float(jerk_mps3))
        if j < 0.8:
            return "Optimal Smoothness", "text-green-500", 1.0
        elif j <= 1.8:
            return "Acceptable Transit", "text-blue-500", 0.75
        elif j <= 2.8:
            return "Slight Discomfort", "text-amber-500", 0.45
        else:
            return "Abrupt Braking", "text-red-500", 0.15
