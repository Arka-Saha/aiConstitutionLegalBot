import os
from flask import Flask, request
import requests
from dotenv import load_dotenv

# This imports your existing RAG function from main.py.
# Make sure rag_answer() ends with "return answer" (see Step 5).
from main import rag_answer

load_dotenv()

app = Flask(__name__)

# Credentials come from your .env file, never hardcoded here.
whatsapp_access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
whatsapp_phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
whatsapp_verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN")

# This is the Meta Graph API endpoint we send replies to.
graph_api_url = f"https://graph.facebook.com/v19.0/{whatsapp_phone_number_id}/messages"


@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """
    Meta calls this once, when you click "Verify and Save" in the dashboard.
    We just need to echo back the challenge value if the verify token matches.
    """
    mode = request.args.get("hub.mode")
    sent_token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and sent_token == whatsapp_verify_token:
        return challenge, 200

    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def receive_whatsapp_message():
    """
    Meta calls this every time a user sends your WhatsApp number a message.
    We pull out the text, run it through the RAG pipeline, and reply.
    """
    incoming_data = request.get_json()

    try:
        entry = incoming_data["entry"][0]
        change = entry["changes"][0]
        value = change["value"]

        # Meta also sends "status" updates (delivered/read) to this same
        # webhook — those don't contain a "messages" key, so skip them.
        if "messages" not in value:
            return "OK", 200

        incoming_message = value["messages"][0]
        sender_number = incoming_message["from"]
        message_text = incoming_message["text"]["body"]

        print(f"Incoming from {sender_number}: {message_text}")

        # This is the same function your terminal version calls.
        answer = rag_answer(message_text)

        send_whatsapp_reply(sender_number, answer)

    except (KeyError, IndexError) as error:
        # Happens if the payload doesn't have the shape we expect
        # (e.g. an unrelated webhook event). Log it and move on.
        print(f"Could not parse webhook payload: {error}")

    # Meta expects a 200 response quickly, regardless of what we did above.
    return "OK", 200


def send_whatsapp_reply(recipient_number, message_text):
    """
    Sends a plain text message back to a WhatsApp user through Meta's API.
    """
    headers = {
        "Authorization": f"Bearer {whatsapp_access_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_number,
        "type": "text",
        "text": {"body": message_text},
    }

    response = requests.post(graph_api_url, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"Failed to send WhatsApp message: {response.status_code} {response.text}")


if __name__ == "__main__":
    print("======================================")
    print(" Indian Legal Aid WhatsApp Bot")
    print(" Running on port 5000")
    print("======================================")
    app.run(port=5000)
