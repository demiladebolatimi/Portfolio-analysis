"""Email service for sending portfolio analysis reports."""

import os
import smtplib
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
from datetime import datetime

import config


def should_send_email(results_df, confidence_threshold=None):
    """Check if email should be sent based on confidence levels."""
    if confidence_threshold is None:
        confidence_threshold = config.EMAIL_CONFIDENCE_THRESHOLD
        
    for idx, row in results_df.iterrows():
        buy_score = row['CurrentBuyScore']
        sell_score = row['CurrentSellScore']
        
        if pd.notna(buy_score) and (buy_score > confidence_threshold or buy_score < (100 - confidence_threshold)):
            return True
        if pd.notna(sell_score) and (sell_score > confidence_threshold or sell_score < (100 - confidence_threshold)):
            return True
    
    return False


def send_email_report(results_df, subject, force_send=False):
    """Send portfolio analysis results and charts via email only if confidence levels are high."""
    try:
        # Check if email should be sent based on confidence levels
        if not force_send and not should_send_email(results_df):
            threshold = config.EMAIL_CONFIDENCE_THRESHOLD
            print(f"📧 Email skipped - No high-confidence signals (>{threshold} or <{100-threshold}) detected")
            return False
        # Load environment variables
        load_dotenv()
        
        # Email configuration
        sender_email = config.EMAIL_SENDER
        receiver_email = config.EMAIL_RECEIVER
        bcc_emails = config.EMAIL_BCC if config.EMAIL_BCC_ENABLED else []
        password = os.getenv('GMAIL_PASSWORD')
        
        if not password:
            # Prompt for password if not in environment
            print("GMAIL_PASSWORD not found in environment variables")
            password = input("Enter your Gmail App Password: ")
            save_to_env = input("Save password to .env file for future use? (y/n): ").lower()
            if save_to_env == 'y':
                with open('.env', 'a') as f:
                    f.write(f"\nGMAIL_PASSWORD={password}")
                print("Password saved to .env file")
        
        print("=" * 60)
        print("EMAIL SETUP")
        print("=" * 60)
        print(f"Sending from: {sender_email}")
        print(f"Sending to: {receiver_email}")
        if bcc_emails:
            print(f"BCC: {len(bcc_emails)} recipient(s) (hidden)")
        else:
            print("BCC: Disabled")
        print("✅ Using saved password from environment variables")
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        
        # Create email body
        body = f"""
        <html>
        <body>
            <h2>Portfolio Analysis Report - {datetime.now().strftime('%Y-%m-%d')}</h2>
            <p>Analysis period: Last 6 months (126 trading days)</p>
            <p>Training data: 5 years historical data</p>
            
            <h3>Summary Results</h3>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <th>Symbol</th>
                    <th>BuyHold</th>
                    <th>Benchmark</th>
                    <th>Manual</th>
                    <th>Q-Learner</th>
                    <th>RandomForest</th>
                    <th>Sell Score</th>
                    <th>Buy Score</th>
                </tr>
        """
        
        # Add results to table
        for idx, row in results_df.iterrows():
            buyhold_val = f"{row['BuyHold']:.2%}" if pd.notna(row['BuyHold']) else 'N/A'
            benchmark_val = f"{row['Benchmark']:.2%}" if pd.notna(row['Benchmark']) else 'N/A'
            manual_val = f"{row['Manual']:.2%}" if pd.notna(row['Manual']) else 'N/A'
            ql_val = f"{row['QLearner']:.2%}" if pd.notna(row['QLearner']) else 'N/A'
            rf_val = f"{row['RandomForest']:.2%}" if pd.notna(row['RandomForest']) else 'N/A'
            sell_score_val = f"{row['CurrentSellScore']:.1f}/100" if pd.notna(row['CurrentSellScore']) else 'N/A'
            buy_score_val = f"{row['CurrentBuyScore']:.1f}/100" if pd.notna(row['CurrentBuyScore']) else 'N/A'
            
            body += f"""
                <tr>
                    <td>{row['Symbol']}</td>
                    <td>{buyhold_val}</td>
                    <td>{benchmark_val}</td>
                    <td>{manual_val}</td>
                    <td>{ql_val}</td>
                    <td>{rf_val}</td>
                    <td>{sell_score_val}</td>
                    <td>{buy_score_val}</td>
                </tr>
            """
        
        body += """
            </table>
            
            <h3>Position Sizing Recommendations</h3>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <th>Symbol</th>
                    <th>Recommendation</th>
                    <th>Trade Size</th>
                    <th>Gradual %</th>
                    <th>Current Price</th>
                </tr>
        """
        
        # Add position sizing recommendations
        for idx, row in results_df.iterrows():
            if pd.notna(row.get('SharesToTrade')) and row['SharesToTrade'] > 0:
                recommendation = row.get('Recommendation', 'HOLD')
                shares_to_trade = row['SharesToTrade']
                trade_action = row.get('TradeAction', 'HOLD')
                gradual_pct = row.get('GradualPercentage', 1.0)
                current_price = row.get('CurrentPrice', 0)
                
                body += f"""
                    <tr>
                        <td>{row['Symbol']}</td>
                        <td>{recommendation}</td>
                        <td>{trade_action} {shares_to_trade}</td>
                        <td>{gradual_pct:.0%}</td>
                        <td>${current_price:.2f}</td>
                    </tr>
                """
        
        body += """
            </table>
            
            <h3>Top 5 Sell Candidates</h3>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <th>Symbol</th>
                    <th>Sell Score</th>
                    <th>6-Month Return</th>
                    <th>Reason</th>
                </tr>
        """
        
        # Get top 5 sell candidates
        top_sells = results_df.nlargest(5, 'CurrentSellScore')
        for idx, row in top_sells.iterrows():
            sell_score = f"{row['CurrentSellScore']:.1f}/100"
            return_val = f"{row['BuyHold']:.2%}" if pd.notna(row['BuyHold']) else 'N/A'
            
            # Generate reason based on scores and performance
            reasons = []
            if row['CurrentSellScore'] > 65:
                reasons.append("High sell score (>65)")
            if row['CurrentBuyScore'] < 35:
                reasons.append("Low buy score (<35)")
            if pd.notna(row['BuyHold']) and row['BuyHold'] > 0.10:
                reasons.append("Strong 6-month performance")
            if pd.notna(row['Manual']) and row['Manual'] < 0:
                reasons.append("Manual strategy underperformance")
            if pd.notna(row['QLearner']) and row['QLearner'] < 0:
                reasons.append("Q-Learner negative signal")
            
            reason = "; ".join(reasons) if reasons else "Technical sell signal"
            
            body += f"""
                <tr>
                    <td>{row['Symbol']}</td>
                    <td>{sell_score}</td>
                    <td>{return_val}</td>
                    <td>{reason}</td>
                </tr>
            """
        
        body += """
            </table>
            
            <h3>Top 5 Buy Candidates</h3>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <th>Symbol</th>
                    <th>Buy Score</th>
                    <th>6-Month Return</th>
                    <th>Reason</th>
                </tr>
        """
        
        # Get top 5 buy candidates
        top_buys = results_df.nlargest(5, 'CurrentBuyScore')
        for idx, row in top_buys.iterrows():
            buy_score = f"{row['CurrentBuyScore']:.1f}/100"
            return_val = f"{row['BuyHold']:.2%}" if pd.notna(row['BuyHold']) else 'N/A'
            
            # Generate reason based on scores and performance
            reasons = []
            if row['CurrentBuyScore'] > 65:
                reasons.append("High buy score (>65)")
            if row['CurrentSellScore'] < 35:
                reasons.append("Low sell score (<35)")
            if pd.notna(row['BuyHold']) and row['BuyHold'] < -0.10:
                reasons.append("Oversold conditions")
            if pd.notna(row['Manual']) and row['Manual'] > 0:
                reasons.append("Manual strategy outperformance")
            if pd.notna(row['QLearner']) and row['QLearner'] > 0:
                reasons.append("Q-Learner positive signal")
            
            reason = "; ".join(reasons) if reasons else "Technical buy signal"
            
            body += f"""
                <tr>
                    <td>{row['Symbol']}</td>
                    <td>{buy_score}</td>
                    <td>{return_val}</td>
                    <td>{reason}</td>
                </tr>
            """
        
        body += """
            </table>
            
            <h3>Key Insights</h3>
            <ul>
                <li>Buy & Hold: Passive portfolio performance</li>
                <li>Benchmark: S&P 500 (SPY) performance</li>
                <li>Manual Strategy: Rule-based trading performance</li>
                <li>Q-Learner: ML-based strategy (Score >65 = SELL, Score <35 = BUY)</li>
                <li>Random Forest: ML model (only if accuracy >55%)</li>
            </ul>
            
            <p>Charts are attached to this email for detailed analysis.</p>
            <p><em>Generated by Portfolio Analysis System</em></p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Attach charts
        chart_files = [
            'portfolio_comparison.png',
            'sell_likelihood_tracking.png',
            'sell_likelihood_buckets.png',
            'sell_likelihood_heatmap.png',
            'buy_likelihood_heatmap.png'
        ]
        
        for chart_file in chart_files:
            if os.path.exists(chart_file):
                with open(chart_file, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {chart_file}'
                    )
                    msg.attach(part)
                    print(f"Attached: {chart_file}")
        
        # Send email
        all_recipients = [receiver_email] + (bcc_emails if bcc_emails else [])
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, all_recipients, msg.as_string())
        server.quit()
        
        print("\n✅ Email sent successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error sending email: {e}")
        import traceback
        traceback.print_exc()
        return False
