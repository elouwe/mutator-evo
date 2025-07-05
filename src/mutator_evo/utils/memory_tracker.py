# src/mutator_evo/utils/memory_tracker.py

import tracemalloc
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class MemoryTracker:
    """Class for tracking memory usage"""
    def __init__(self, interval=60):
        self.interval = interval
        self.snapshots = []
        self.running = False
        self._stats = defaultdict(list)
    
    def start(self):
        """Start memory tracking"""
        if not self.running:
            tracemalloc.start()
            self.running = True
            logger.info("Memory tracking started")
    
    def stop(self):
        """Stop memory tracking"""
        if self.running:
            tracemalloc.stop()
            self.running = False
            logger.info("Memory tracking stopped")
    
    def take_snapshot(self, label):
        """Take a snapshot of current memory state"""
        if self.running:
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append((label, snapshot))
            logger.info(f"Memory snapshot taken: {label}")
    
    def generate_report(self, filename="memory_report.txt"):
        """Generate memory usage report"""
        if not self.snapshots:
            return
        
        report = ["Memory Usage Report\n" + "="*30]
        
        for i in range(1, len(self.snapshots)):
            label, snapshot = self.snapshots[i]
            prev_label, prev_snapshot = self.snapshots[i-1]
            
            top_stats = snapshot.compare_to(prev_snapshot, 'lineno')
            
            report.append(f"\nChanges from {prev_label} to {label}:")
            for stat in top_stats[:10]:
                report.append(str(stat))
        
        with open(filename, "w") as f:
            f.write("\n".join(report))
        
        logger.info(f"Memory report generated: {filename}")