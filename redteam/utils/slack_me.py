import json
import traceback

import urllib3

# Personal chat webhook - TODO: Add to bashrc
# webhook_url = 'https://hooks.slack.com/services/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'
webhook_url = "https://hooks.slack.com/services/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"
# # Channel url
# webhook_url_channel = ''


# Send Slack notification based on the given message
def slack_notification(message):
    try:
        slack_message = {"text": message}

        http = urllib3.PoolManager()
        response = http.request(
            "POST",
            webhook_url,
            body=json.dumps(slack_message),
            headers={"Content-Type": "application/json"},
            retries=False,
        )
    except:
        traceback.print_exc()

    return True
