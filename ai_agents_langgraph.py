import os
import re
import imaplib
import email
import yagmail
from email.header import decode_header
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId

from langchain.tools import tool
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()
app_pass=os.getenv('GOOGLE_APP_PASSWORD')
mc=os.getenv('MONGO_CLIENT')
client = MongoClient(mc)
db = client['pothole_app']

MONGODB_URI = mc
DB_NAME = "pothole_app"
COMPLAINT_COLLECTION = "complaints"
RESOLVED_COLLECTION = "resolved_complaints"

complaints_collection = db["complaints"]
resolved_complaints_collection = db["resolved_complaints"]
SENDER_EMAIL = "mohitchauhan22334@gmail.com"
APP_PASSWORD = app_pass
TARGET_EMAIL = "kkjj1234560@gmail.com"

complaints_collection = db[COMPLAINT_COLLECTION]
resolved_collection = db[RESOLVED_COLLECTION]

class AgentState(TypedDict):
    template: str
    status: str
    skip_email: bool

def decode_mime_words(s):
    """Decode MIME-encoded subject line."""
    decoded_parts = decode_header(s)
    return ''.join(
        part.decode(enc or 'utf-8') if isinstance(part, bytes) else part
        for part, enc in decoded_parts
    )
    
def prepare_email_body(complaint, template):
    return template.format(
        name=complaint['user_email'].split('@')[0],
        email=complaint['user_email'],
        contact=complaint['contact'],
        address=f"Lat: {complaint['latitude']}, Lon: {complaint['longitude']}"
    )

@tool
def send_real_email(subject: str, body: str) -> str:
    """Send a formal email with subject and body to the target Gmail using Yagmail."""
    print("📤 [Agent] send_real_email triggered")
    try:
        yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)
        yag.send(to=TARGET_EMAIL, subject=subject, contents=body)
        return "sent"
    except Exception as e:
        return f"Email failed: {e}"

@tool
def check_reply_and_resolve(_: str = "", max_emails: int = 10) -> str:
    """Checks up to `max_emails` unread emails for resolution keywords and resolves matching complaints."""
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(SENDER_EMAIL, APP_PASSWORD)
        mail.select("inbox")

     
        status, data = mail.search(None, 'UNSEEN')
        email_ids = data[0].split()

        if not email_ids:
            return "📭 No new unread emails."

        email_ids = email_ids[-max_emails:]

        resolution_keywords = ["resolved", "done", "fixed", "completed", "solved"]
        resolved_count = 0

        for eid in reversed(email_ids): 
            _, msg_data = mail.fetch(eid, '(RFC822)')
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            subject = decode_mime_words(msg.get("subject", ""))
            from_email = msg.get("from", "")


            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain" and not part.get("Content-Disposition"):
                        charset = part.get_content_charset() or "utf-8"
                        body += part.get_payload(decode=True).decode(charset, errors="replace")
            else:
                charset = msg.get_content_charset() or "utf-8"
                body = msg.get_payload(decode=True).decode(charset, errors="replace")

        
            if any(keyword in body.lower() for keyword in resolution_keywords):
                match = re.search(r"Complaint\s+#([a-f0-9]{24})", subject)
                if match:
                    complaint_id = match.group(1)
                    complaint = complaints_collection.find_one({"_id": ObjectId(complaint_id)})

                    if complaint:
                        complaint["status"] = "Resolved"
                        complaint["resolution_date"] = datetime.now()
                        resolved_complaints_collection.insert_one(complaint)
                        complaints_collection.delete_one({"_id": ObjectId(complaint_id)})
                        resolved_count += 1
                        print(f" Complaint {complaint_id} marked as resolved.")
                    else:
                        print(f" Complaint ID {complaint_id} not found in DB.")
                else:
                    print(f" No complaint ID found in subject: {subject}")
            else:
                print(f" No resolution keywords in email from {from_email}")

        return f" {resolved_count} complaints resolved from email replies."

    except Exception as e:
        return f" Email check failed: {e}"

def build_langgraph_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    def fetch_and_format(state):
        if state.get("skip_email", False):
            return {**state, "status": "Email skipped"}
        
        complaint = complaints_collection.find_one({
            "complaint_approved_by_admin": False,
            "validated_by_model": 1
        })
        if not complaint:
            return {**state, "status": "No complaints found"}

        complaint_id = str(complaint["_id"])
        complaint["id"] = complaint_id
        formatted = prepare_email_body(complaint, state["template"])
        subject = f"Pothole Complaint #{complaint_id}"

        send_real_email.invoke({
            "subject": f"Pothole Complaint #{complaint_id}",
            "body": formatted
    })

        complaints_collection.update_one(
            {"_id": ObjectId(complaint_id)},
            {"$set": {
                "complaint_approved_by_admin": True,
                "status": "Approved by Admin",
                "approval_date": datetime.now()
            }}
        )
        return {**state, "complaint_id": complaint_id, "status": "Email sent"}

    def check_reply_node(state):
       result = check_reply_and_resolve.invoke({"max_emails": 10})
       return {**state, "reply_check": result}

    graph = StateGraph(AgentState)
    graph.add_node("process_complaint", fetch_and_format)
    graph.add_node("check_reply", check_reply_node)
    graph.set_entry_point("process_complaint")
    graph.add_edge("process_complaint", "check_reply")
    graph.set_finish_point("check_reply")

    return graph.compile()

agent_executor = build_langgraph_agent()

