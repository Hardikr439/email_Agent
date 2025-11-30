"""
Simple Email Sending Service
Sends emails via Brevo API
"""

import os
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

# Optional: LangChain + Gemini for subject/body enhancement
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import ChatPromptTemplate
except ImportError:  # Keep service working even if LangChain/Gemini is not installed
    print("Not working")
    ChatGoogleGenerativeAI = None
    ChatPromptTemplate = None


class ServiceResult:
    """Simple result object for email sending"""
    def __init__(self, recipient: str, subject: str, success: bool, message: str = ""):
        self.recipient = recipient
        self.subject = subject
        self.success = success
        self.message = message
        self.raw = f"Email {'sent' if success else 'failed'} to {recipient}: {message}"
        self.json_dict = {
            "recipient": recipient,
            "subject": subject,
            "success": success,
            "message": message,
            "task": "send_email"
        }


class EmailService:
    """Service for sending emails"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.brevo_api_key = os.getenv("BREVO_API_KEY", "")
        self.sender_email = os.getenv("SENDER_EMAIL", "")
        self.sender_name = os.getenv("SENDER_NAME", "Email Service")

        # Gemini / LangChain configuration (optional)
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
        
        # Initialize Brevo API configuration
        if self.brevo_api_key:
            configuration = sib_api_v3_sdk.Configuration()
            configuration.api_key['api-key'] = self.brevo_api_key
            self.api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
        else:
            self.api_instance = None

    def _enhance_email_with_gemini(self, keypoints, original_subject: str, original_body: str):
        """Use Gemini via LangChain to turn keypoints into a subject and body.

        This is optional and will gracefully fall back to the original
        subject/body if Gemini or LangChain are not configured.
        """

        # Guard: LangChain or Gemini not available
        if not ChatGoogleGenerativeAI or not ChatPromptTemplate:
            if self.logger:
                self.logger.info("LangChain / Gemini not installed; skipping email enhancement.")
            return original_subject, original_body

        if not self.gemini_api_key:
            if self.logger:
                self.logger.info("GOOGLE_API_KEY not set; skipping email enhancement.")
            return original_subject, original_body

        try:
            if self.logger:
                self.logger.info("Enhancing email with Gemini from keypoints.")

            llm = ChatGoogleGenerativeAI(
                model=self.gemini_model_name,
                google_api_key=self.gemini_api_key,
                temperature=0.7,
            )

            prompt = ChatPromptTemplate.from_template(
                (
                    "You are an expert email copywriter. Write concise, impactful, "
                    "and professional emails that clearly achieve the sender's goal.\n\n"
                    "You are given:\n"
                    "- KEYPOINTS: a plain‑language description of what the email should do\n"
                    "- EXISTING SUBJECT/BODY: an earlier draft which may be weak, unclear, or redundant\n\n"
                    "Your tasks:\n"
                    "1) **Understand the intent** from KEYPOINTS (purpose, audience, desired outcome, tone).\n"
                    "2) **Improve**, do NOT copy: significantly improve on the existing subject/body.\n"
                    "   - You may reuse important facts or phrases, but avoid mirroring the old wording.\n"
                    "3) Write:\n"
                    "   - A compelling, specific subject line (max 70 characters).\n"
                    "   - A plain‑text email body with 3–6 short paragraphs.\n"
                    "     * Friendly, confident, and professional.\n"
                    "     * Clear call‑to‑action if appropriate.\n"
                    "     * No markdown, no bullet points, no code blocks.\n\n"
                    "Formatting rules (IMPORTANT):\n"
                    "- Respond with **ONLY** valid JSON.\n"
                    "- Use exactly these keys:\n"
                    "  {{\n"
                    '    "subject": "<final subject line as a string>",\n'
                    '    "body": "<final email body as a single string>"\n'
                    "  }}\n"
                    "- Do NOT include any explanations, comments, or additional fields.\n\n"
                    "KEYPOINTS (what this email should do):\n"
                    "{keypoints}\n\n"
                    "EXISTING SUBJECT (may be empty or low quality):\n"
                    "{subject}\n\n"
                    "EXISTING BODY (may be empty or low quality):\n"
                    "{body}\n"
                )
            )

            chain = prompt | llm
            response = chain.invoke(
                {
                    # Treat keypoints as a single description string
                    "keypoints": "\n".join(keypoints),
                    "subject": original_subject,
                    "body": original_body,
                }
            )

            text = getattr(response, "content", str(response))

            import json

            # Be resilient to extra text by extracting the first JSON object
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                import re
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if not match:
                    raise
                data = json.loads(match.group(0))

            new_subject = (data.get("subject") or original_subject or "").strip()
            new_body = (data.get("body") or original_body or "").strip()

            if self.logger:
                self.logger.info("Gemini enhancement completed.")

            return new_subject, new_body

        except Exception as e:
            if self.logger:
                self.logger.error(f"Gemini enhancement failed, using original subject/body: {e}")
            return original_subject, original_body
    
    async def execute_task(self, input_data: dict) -> ServiceResult:
        """
        Send email
        
        Args:
            input_data: Dictionary containing:
                - 'recipient_email': Email address to send to
                - 'subject': Email subject
                - 'body': Email body content
                
        Returns:
            ServiceResult with email sending status
        """
        recipient_email = input_data.get("recipient_email", "")
        subject = input_data.get("subject", "")
        body = input_data.get("body", "")
        
        if self.logger:
            self.logger.info(f"Sending email to {recipient_email} with subject: '{subject}'")

        # Always allow Gemini to enhance, using the current subject/body
        # as the "keypoints" for synthesis. If Gemini is not configured,
        # _enhance_email_with_gemini will just return the originals.
        subject, body = self._enhance_email_with_gemini(
            keypoints=[subject, body],
            original_subject=subject,
            original_body=body,
        )
        
        # Validate inputs
        if not recipient_email:
            error_msg = "recipient_email is required"
            if self.logger:
                self.logger.error(error_msg)
            return ServiceResult(recipient_email, subject, False, error_msg)
        
        if not subject:
            error_msg = "subject is required"
            if self.logger:
                self.logger.error(error_msg)
            return ServiceResult(recipient_email, subject, False, error_msg)
        
        if not body:
            error_msg = "body is required"
            if self.logger:
                self.logger.error(error_msg)
            return ServiceResult(recipient_email, subject, False, error_msg)
        
        if not self.brevo_api_key:
            error_msg = "Brevo API key not configured (BREVO_API_KEY missing)"
            if self.logger:
                self.logger.error(error_msg)
            return ServiceResult(recipient_email, subject, False, error_msg)
        
        if not self.sender_email:
            error_msg = "Sender email not configured (SENDER_EMAIL missing)"
            if self.logger:
                self.logger.error(error_msg)
            return ServiceResult(recipient_email, subject, False, error_msg)
        
        try:
            # Create Brevo email object
            sender = {"name": self.sender_name, "email": self.sender_email}
            to = [{"email": recipient_email}]
            
            send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
                to=to,
                sender=sender,
                subject=subject,
                text_content=body
            )
            
            # Send email via Brevo API
            api_response = self.api_instance.send_transac_email(send_smtp_email)
            
            success_msg = f"Email sent successfully to {recipient_email} (Message ID: {api_response.message_id})"
            if self.logger:
                self.logger.info(success_msg)
            
            return ServiceResult(recipient_email, subject, True, success_msg)
        
        except ApiException as e:
            error_msg = f"Brevo API error: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            return ServiceResult(recipient_email, subject, False, error_msg)
        
        except Exception as e:
            error_msg = f"Error sending email: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            return ServiceResult(recipient_email, subject, False, error_msg)


def get_agentic_service(logger=None):
    """
    Factory function to get the email service
    """
    return EmailService(logger)
