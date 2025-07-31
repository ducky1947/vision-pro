# email_sender.py
import os
import smtplib
import ssl
import datetime
import numpy as np
import cv2
import threading # For sending emails in a separate thread

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import config # Import the centralized configuration

def _send_email_actual(image_path, timestamp, camera_id=""):
    """
    Internal function to send an email alert with intruder image attachment.
    This function performs the actual SMTP communication.
    It should be called within a separate thread to avoid blocking.
    """
    email_config = config.EMAIL_CONFIG # Get email configuration from the config module

    # Basic validation for email configuration
    if not all(key in email_config and email_config[key] for key in ['sender', 'password', 'receiver']):
        print("DEBUG: email_sender: Email alerts disabled - incomplete or missing configuration.", flush=True)
        return False # Indicate failure due to config

    print(f"DEBUG: email_sender: Preparing email alert for {image_path} (Camera ID: {camera_id})", flush=True)
    
    msg = MIMEMultipart()
    msg['From'] = email_config['sender']
    msg['To'] = email_config['receiver']
    msg['Subject'] = email_config['subject']
    
    # Email body with HTML formatting for better presentation
    body = f"""
    <html>
    <head></head>
    <body>
        <h2>Intruder Detected!</h2>
        <p><b>Time:</b> {timestamp}</p>
        <p><b>Camera ID:</b> {camera_id}</p>
        <p>The face recognition system detected an unrecognized face.</p>
        <p>Please review the attached image for details.</p>
        <br>
        <p>This is an automated alert from your surveillance system.</p>
    </body>
    </html>
    """
    msg.attach(MIMEText(body, 'html'))
    
    # Attach the intruder image if path is valid
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as f:
                img_data = f.read()
            image = MIMEImage(img_data, name=os.path.basename(image_path))
            msg.attach(image)
            print(f"DEBUG: email_sender: Attached image: {os.path.basename(image_path)}", flush=True)
        except Exception as e:
            print(f"ERROR: email_sender: Failed to attach image {image_path}: {e}", flush=True)
            # Decide if you want to send email without image or abort. For now, continue without image.
    else:
        print(f"WARNING: email_sender: Image file '{image_path}' not found or path invalid. Email will be sent without attachment.", flush=True)

    try:
        # Secure connection to SMTP server using TLS
        context = ssl.create_default_context()
        print(f"DEBUG: email_sender: Connecting to SMTP server {email_config['smtp_server']}:{email_config['smtp_port']}...", flush=True)
        with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
            server.starttls(context=context) # Upgrade the connection to a secure encrypted SSL/TLS connection
            server.login(email_config['sender'], email_config['password']) # Login to the SMTP server
            server.send_message(msg) # Send the constructed email message
        
        print(f"DEBUG: email_sender: Email alert sent successfully to {email_config['receiver']}.", flush=True)
        return True # Indicate success
    except smtplib.SMTPAuthenticationError as e:
        print(f"CRITICAL ERROR: email_sender: SMTP Authentication Failed. Check sender email/password in config.py. Error: {e}", flush=True)
        return False
    except smtplib.SMTPConnectError as e:
        print(f"CRITICAL ERROR: email_sender: SMTP Connection Failed. Check SMTP server/port or network. Error: {e}", flush=True)
        return False
    except smtplib.SMTPException as e:
        print(f"CRITICAL ERROR: email_sender: An SMTP error occurred: {e}", flush=True)
        return False
    except Exception as e:
        print(f"CRITICAL ERROR: email_sender: An unexpected error occurred while sending email: {e}", flush=True)
        return False

def send_alert_in_thread(image_path, timestamp, camera_id=""):
    """
    Creates and starts a new daemon thread to send an email alert.
    This function should be called from the main process or another thread
    when an email needs to be sent without blocking.
    """
    print(f"DEBUG: email_sender: Starting new thread to send email alert for {image_path}.", flush=True)
    # daemon=True ensures the thread will exit automatically when the main program exits
    email_thread = threading.Thread(
        target=_send_email_actual,
        args=(image_path, timestamp, camera_id),
        daemon=True
    )
    email_thread.start()

# Optional: Standalone test for this module
if __name__ == "__main__":
    print("\n--- Testing email_sender.py (Standalone) ---", flush=True)

    # --- Configuration Check ---
    if not all(key in config.EMAIL_CONFIG and config.EMAIL_CONFIG[key] for key in ['sender', 'password', 'receiver']):
        print("WARNING: Email configuration incomplete in config.py. Cannot run email send test.", flush=True)
        print("Please set 'sender', 'password', and 'receiver' in config.py for email tests.", flush=True)
    else:
        # Create a dummy image for testing attachment
        dummy_image_path = "test_intruder_image.jpg"
        try:
            # Create a simple black image using OpenCV
            import cv2
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8) # 100x100 black image
            cv2.imwrite(dummy_image_path, dummy_image)
            print(f"DEBUG: Created dummy image for testing: {dummy_image_path}", flush=True)
        except ImportError:
            print("WARNING: OpenCV not found. Cannot create dummy image. Email will be tested without attachment.", flush=True)
            dummy_image_path = None
        except Exception as e:
            print(f"ERROR: Failed to create dummy image: {e}. Email will be tested without attachment.", flush=True)
            dummy_image_path = None

        # Test sending an alert
        test_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        test_camera_id = "TEST_CAM_001"
        print(f"\nAttempting to send a test email alert to {config.EMAIL_CONFIG['receiver']}...", flush=True)
        send_alert_in_thread(dummy_image_path, test_timestamp, test_camera_id)
        
        # Give some time for the thread to run and send the email
        print("Waiting 5 seconds for email thread to attempt sending...", flush=True)
        import time
        time.sleep(5) 
        print("Check your receiver inbox for the test email. Also check spam/junk folder.", flush=True)

        # Clean up dummy image
        if dummy_image_path and os.path.exists(dummy_image_path):
            try:
                os.remove(dummy_image_path)
                print(f"DEBUG: Cleaned up dummy image: {dummy_image_path}", flush=True)
            except Exception as e:
                print(f"ERROR: Failed to delete dummy image: {e}", flush=True)

    print("\n--- email_sender.py Standalone Test Complete ---", flush=True)