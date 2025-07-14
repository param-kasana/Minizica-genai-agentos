import asyncio
from typing import Annotated, List, Optional, Union
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext
import smtplib
from email.message import EmailMessage

AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NDA0ZTZhOC04MDUyLTRlODQtYTc1NS05ZmI2NjVkZTU3ODIiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImI5OWQ4YTNmLWFjNTItNGRmNC05YjExLTYxNzljZDQ4Y2MxNSJ9.uIlNn5KC5q-c65OCAkWAHN0QHMpa2-5CIVhEnWIwOXE"  # noqa: E501
session = GenAISession(jwt_token=AGENT_JWT)

def send_email(
    to_emails: Union[str, List[str]],
    cc_emails: Optional[Union[str, List[str]]] = None,
    bcc_emails: Optional[Union[str, List[str]]] = None,
    subject: str = '',
    body: str = ''
):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = "visionversetest@gmail.com"
    smtp_password = "mjvqutjfqggkxukj"
    from_email = "visionversetest@gmail.com"

    def normalize(addresses):
        if not addresses:
            return []
        if isinstance(addresses, str):
            return [a.strip() for a in addresses.replace(';', ',').split(',') if a.strip()]
        return addresses

    to_emails = normalize(to_emails)
    cc_emails = normalize(cc_emails)
    bcc_emails = normalize(bcc_emails)

    msg = EmailMessage()
    msg['From'] = from_email
    msg['To'] = ', '.join(to_emails)
    if cc_emails:
        msg['Cc'] = ', '.join(cc_emails)
    msg['Subject'] = subject
    msg.set_content(body)

    recipients = to_emails + cc_emails + bcc_emails

    with smtplib.SMTP(smtp_server, smtp_port) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(smtp_username, smtp_password)
        smtp.send_message(msg, from_addr=from_email, to_addrs=recipients)

@session.bind(
    name="email_agent",
    description="Email related functions."
)
async def email_agent(
    agent_context: GenAIContext,
    to_emails: Annotated[
        Union[str, List[str]],
        "Recipient email address(es), comma or semicolon separated or list."
    ],
    subject: Annotated[
        str,
        "Subject of the email."
    ],
    body: Annotated[
        str,
        "Body of the email."
    ],
    cc_emails: Annotated[
        Optional[Union[str, List[str]]],
        "CC email address(es), optional, comma or semicolon separated or list."
    ] = None,
    bcc_emails: Annotated[
        Optional[Union[str, List[str]]],
        "BCC email address(es), optional, comma or semicolon separated or list."
    ] = None,
):
    """Send an email using Gmail SMTP."""
    try:
        send_email(
            to_emails=to_emails,
            cc_emails=cc_emails,
            bcc_emails=bcc_emails,
            subject=subject,
            body=body
        )
        return "Email sent successfully."
    except Exception as e:
        return f"Failed to send email: {e}"

async def main():
    print(f"Agent with token '{AGENT_JWT}' started")
    await session.process_events()

if __name__ == "__main__":
    asyncio.run(main())
