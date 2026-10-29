import time
import numpy as np

class CAMMessage:
    """
    Cooperative Awareness Message (CAM) - ETSI EN 302 637-2 Standard
    Periodic beacon broadcast at 10 Hz for mutual situational awareness.
    """
    def __init__(self, station_id, position, speed, heading, acceleration):
        self.message_type = "CAM"
        self.station_id = station_id
        self.timestamp = time.time()
        self.position = position  # (x, y) in meters
        self.speed = float(speed)  # m/s
        self.heading = float(heading)  # radians
        self.acceleration = float(acceleration)  # m/s^2


class DENMMessage:
    """
    Decentralized Environmental Notification Message (DENM) - ETSI EN 302 637-3 Standard
    Event-triggered emergency alert broadcast upon safety hazard / congestion / accident detection.
    """
    def __init__(self, action_id, station_id, cause_code, event_position, relevance_radius=450.0):
        self.message_type = "DENM"
        self.action_id = action_id
        self.station_id = station_id
        self.timestamp = time.time()
        self.cause_code = cause_code  # "ACCIDENT", "EMERGENCY_BRAKE", "ROAD_BLOCKAGE"
        self.event_position = event_position  # (x, y)
        self.relevance_radius = float(relevance_radius)  # meters
        self.expiry_time = self.timestamp + 30.0  # valid for 30s


class V2XCommunicationNetwork:
    """
    Simulated IEEE 802.11p / C-V2X (Cellular V2X) Mesh Communication Layer.
    Handles message dissemination, transmission range checks, latency, and packet delivery ratio (PDR).
    """
    def __init__(self, comm_range=500.0, avg_latency_ms=8.5, packet_loss_rate=0.015):
        self.comm_range = comm_range
        self.avg_latency_ms = avg_latency_ms
        self.packet_loss_rate = packet_loss_rate
        self.message_queue = []
        self.active_denm_alerts = []
        self.total_packets_sent = 0
        self.total_packets_delivered = 0

    def broadcast_cam(self, sender_id, position, speed, heading, acceleration):
        """Broadcasts periodic CAM beacon"""
        self.total_packets_sent += 1
        if np.random.rand() > self.packet_loss_rate:
            self.total_packets_delivered += 1
            cam = CAMMessage(sender_id, position, speed, heading, acceleration)
            self.message_queue.append(cam)
            return cam
        return None

    def broadcast_denm(self, sender_id, cause_code, event_position, relevance_radius=450.0):
        """Broadcasts high-priority emergency DENM packet"""
        self.total_packets_sent += 1
        denm = DENMMessage(
            action_id=f"DENM_{len(self.active_denm_alerts)+1}",
            station_id=sender_id,
            cause_code=cause_code,
            event_position=event_position,
            relevance_radius=relevance_radius
        )
        self.active_denm_alerts.append(denm)
        self.message_queue.append(denm)
        self.total_packets_delivered += 1
        return denm

    def get_alerts_in_range(self, vehicle_position):
        """Returns active DENM hazard alerts within the vehicle's V2X reception radius"""
        relevant_alerts = []
        now = time.time()
        
        # Filter expired alerts
        self.active_denm_alerts = [a for a in self.active_denm_alerts if a.expiry_time > now]
        
        for alert in self.active_denm_alerts:
            dist = np.hypot(vehicle_position[0] - alert.event_position[0], vehicle_position[1] - alert.event_position[1])
            if dist <= alert.relevance_radius:
                relevant_alerts.append(alert)
                
        return relevant_alerts

    def get_pdr(self):
        """Returns empirical Packet Delivery Ratio (PDR)"""
        if self.total_packets_sent == 0:
            return 100.0
        return (self.total_packets_delivered / self.total_packets_sent) * 100.0
