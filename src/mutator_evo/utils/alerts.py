# src/mutator_evo/utils/alerts.py
import logging
import os
from slack_sdk import WebhookClient

logger = logging.getLogger(__name__)

class AlertManager:
    """Class for sending alerts to Slack"""
    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url or os.getenv('SLACK_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        
        if self.enabled:
            logger.info("Slack alerting enabled")
        else:
            logger.warning("Slack webhook URL not set, alerting disabled")
    
    def send_alert(self, message):
        if not self.enabled:
            return False
            
        try:
            client = WebhookClient(self.webhook_url)
            response = client.send(text=message)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            return False
    
    def send_degradation_alert(self, generation, scores):
        if not self.enabled:
            return False
            
        message = (
            f":warning: *Strategy Degradation Alert* :warning:\n"
            f"Generation: {generation}\n"
            f"Best scores: {scores[-3:]}\n"
            f"Degradation detected for 3 consecutive generations!"
        )
        return self.send_alert(message)
    