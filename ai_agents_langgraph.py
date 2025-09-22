# Updated Flask integration - Replace your existing agent code with this

import os
import re
import imaplib
import email
import yagmail
from email.header import decode_header
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from flask import current_app

from langchain.tools import tool
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

# Configuration
app_pass = os.getenv('GOOGLE_APP_PASSWORD')
mc = os.getenv('MONGO_CLIENT')
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

# Global variables
executor = ThreadPoolExecutor(max_workers=2)
agent_executor = None
task_manager = None

class AgentState(TypedDict):
    template: str
    status: str
    skip_email: bool

def decode_mime_words(s):
    """Decode MIME-encoded subject line."""
    try:
        decoded_parts = decode_header(s)
        return ''.join(
            part.decode(enc or 'utf-8') if isinstance(part, bytes) else part
            for part, enc in decoded_parts
        )
    except Exception as e:
        print(f"Error decoding MIME words: {e}")
        return str(s)

def prepare_email_body(complaint, template):
    """Prepare email body with error handling."""
    try:
        return template.format(
            name=complaint.get('user_email', 'Unknown').split('@')[0],
            email=complaint.get('user_email', 'Unknown'),
            contact=complaint.get('contact', 'Unknown'),
            address=f"Lat: {complaint.get('latitude', 'N/A')}, Lon: {complaint.get('longitude', 'N/A')}"
        )
    except Exception as e:
        print(f"Error preparing email body: {e}")
        return "Error preparing complaint details."

@tool
def send_real_email(subject: str, body: str) -> str:
    """Send email with timeout and error handling."""
    print("📤 [Agent] send_real_email triggered")
    try:
        yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)
        yag.send(to=TARGET_EMAIL, subject=subject, contents=body)
        yag.close()
        print(f"✅ Email sent successfully: {subject}")
        return "sent"
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return f"Email failed: {str(e)}"

@tool
def check_reply_and_resolve(max_emails: int = 3) -> str:
    """Check emails with reduced batch size and timeout."""
    print("📧 [Agent] Checking for email replies...")
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(SENDER_EMAIL, APP_PASSWORD)
        mail.select("inbox")

        status, data = mail.search(None, 'UNSEEN')
        email_ids = data[0].split()

        if not email_ids:
            mail.close()
            mail.logout()
            return "📭 No new unread emails."

        email_ids = email_ids[-max_emails:]
        resolution_keywords = ["resolved", "done", "fixed", "completed", "solved"]
        resolved_count = 0

        for eid in reversed(email_ids):
            try:
                _, msg_data = mail.fetch(eid, '(RFC822)')
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)

                subject = decode_mime_words(msg.get("subject", ""))
                body = extract_email_body(msg, max_size=500)

                if any(keyword in body.lower() for keyword in resolution_keywords):
                    match = re.search(r"Complaint\s+#([a-f0-9]{24})", subject)
                    if match:
                        complaint_id = match.group(1)
                        if resolve_complaint(complaint_id):
                            resolved_count += 1

            except Exception as e:
                print(f"Error processing email {eid}: {e}")
                continue

        mail.close()
        mail.logout()
        return f"✅ {resolved_count} complaints resolved from email replies."

    except Exception as e:
        print(f"❌ Email check failed: {e}")
        return f"Email check failed: {str(e)}"

def extract_email_body(msg, max_size=500):
    """Extract email body with size limit."""
    body = ""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if (part.get_content_type() == "text/plain" and 
                    not part.get("Content-Disposition")):
                    charset = part.get_content_charset() or "utf-8"
                    content = part.get_payload(decode=True).decode(charset, errors="replace")
                    body += content[:max_size]
                    break
        else:
            charset = msg.get_content_charset() or "utf-8"
            body = msg.get_payload(decode=True).decode(charset, errors="replace")[:max_size]
    except Exception as e:
        print(f"Error extracting body: {e}")
        body = ""
    return body

def resolve_complaint(complaint_id):
    """Resolve complaint in database."""
    try:
        complaint = complaints_collection.find_one({"_id": ObjectId(complaint_id)})
        if complaint:
            complaint["status"] = "Resolved"
            complaint["resolution_date"] = datetime.now()
            resolved_complaints_collection.insert_one(complaint)
            complaints_collection.delete_one({"_id": ObjectId(complaint_id)})
            print(f"✅ Complaint {complaint_id} marked as resolved.")
            return True
    except Exception as e:
        print(f"❌ Error resolving complaint {complaint_id}: {e}")
    return False

def build_lightweight_agent():
    """Build a lightweight agent optimized for Render."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=100
        )

        def fetch_and_format(state):
            if state.get("skip_email", False):
                return {**state, "status": "Email skipped"}
            
            try:
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

                result = send_real_email.invoke({
                    "subject": subject,
                    "body": formatted
                })

                if result == "sent":
                    complaints_collection.update_one(
                        {"_id": ObjectId(complaint_id)},
                        {"$set": {
                            "complaint_approved_by_admin": True,
                            "status": "Approved by Admin",
                            "approval_date": datetime.now()
                        }}
                    )
                    return {**state, "complaint_id": complaint_id, "status": "Email sent successfully"}
                else:
                    return {**state, "status": f"Email failed: {result}"}

            except Exception as e:
                print(f"❌ Error in fetch_and_format: {e}")
                return {**state, "status": f"Error: {str(e)}"}

        def check_reply_node(state):
            try:
                result = check_reply_and_resolve.invoke({"max_emails": 3})
                return {**state, "reply_check": result}
            except Exception as e:
                print(f"❌ Error in check_reply_node: {e}")
                return {**state, "reply_check": f"Error: {str(e)}"}

        graph = StateGraph(AgentState)
        graph.add_node("process_complaint", fetch_and_format)
        graph.add_node("check_reply", check_reply_node)
        graph.set_entry_point("process_complaint")
        graph.add_edge("process_complaint", "check_reply")
        graph.set_finish_point("check_reply")

        return graph.compile()

    except Exception as e:
        print(f"❌ Error building agent: {e}")
        return None

class BackgroundTaskManager:
    def __init__(self):
        self.running = False
        self.thread = None
    
    def start_background_tasks(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._background_loop, daemon=True)
            self.thread.start()
            print("🚀 Background task manager started")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def _background_loop(self):
        while self.running:
            try:
                time.sleep(300)  # 5 minutes
                print("🔄 Running background email check...")
                
                if agent_executor:
                    future = executor.submit(self._run_agent_check)
                    try:
                        future.result(timeout=30)
                    except Exception as e:
                        print(f"❌ Background task timeout/error: {e}")
                
            except Exception as e:
                print(f"❌ Background loop error: {e}")
                time.sleep(60)
    
    def _run_agent_check(self):
        try:
            result = agent_executor.invoke({
                "template": "",
                "skip_email": True
            })
            print(f"✅ Background check completed: {result.get('reply_check', 'Done')}")
        except Exception as e:
            print(f"❌ Agent check error: {e}")

def initialize_agent_system():
    """Initialize the agent system."""
    global agent_executor, task_manager
    
    try:
        print("🚀 Initializing agent system...")
        
        # Build agent
        agent_executor = build_lightweight_agent()
        if not agent_executor:
            print("❌ Failed to build agent")
            return False
        
        # Start background tasks
        task_manager = BackgroundTaskManager()
        task_manager.start_background_tasks()
        
        print("✅ Agent system initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize agent system: {e}")
        return False

def trigger_manual_agent(template):
    """Trigger agent manually with timeout protection."""
    if not agent_executor:
        return {"status": "Agent not initialized", "success": False}
    
    try:
        future = executor.submit(agent_executor.invoke, {
            "template": template,
            "skip_email": False
        })
        
        result = future.result(timeout=45)
        return {"status": result.get("status", "completed"), "success": True}
        
    except Exception as e:
        print(f"❌ Manual agent trigger failed: {e}")
        return {"status": f"Error: {str(e)}", "success": False}

def cleanup_agent_system():
    """Cleanup agent system resources."""
    global task_manager
    if task_manager:
        task_manager.stop()
    executor.shutdown(wait=False)
    print("🧹 Agent system cleanup completed")
