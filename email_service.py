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


def should_send_email(results_df, confidence_threshold=65):
    """Check if email should be sent based on confidence levels."""
    for idx, row in results_df.iterrows():
        buy_score = row['CurrentBuyScore']
        sell_score = row['CurrentSellScore']
        
        if pd.notna(buy_score) and (buy_score > confidence_threshold or buy_score < (100 - confidence_threshold)):
            return True
        if pd.notna(sell_score) and (sell_score > confidence_threshold or sell_score < (100 - confidence_threshold)):
            return True
    
    return False


def send_email_report(results_df, subject, force_send=False, market_regime="NEUTRAL"):
    """Send portfolio analysis results and charts via email only if confidence levels are high."""
    try:
        # Check if email should be sent based on confidence levels
        if not force_send and not should_send_email(results_df):
            print("📧 Email skipped - No high-confidence signals (>65 or <35) detected")
            return False
        # Load environment variables
        load_dotenv()
        
        # Email configuration
        sender_email = config.EMAIL_SENDER
        receiver_email = config.EMAIL_RECEIVER
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
            <p><strong>Market Regime: {market_regime}</strong></p>
            
            <h3>Summary Results</h3>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <th>Symbol</th>
                    <th>Shares</th>
                    <th>BuyHold</th>
                    <th>Benchmark</th>
                    <th>Manual</th>
                    <th>Q-Learner</th>
                    <th>RandomForest</th>
                    <th>CurrentSellScore</th>
                    <th>CurrentBuyScore</th>
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
                    <td>{row['Shares']}</td>
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
            
            <h3>Position Sizing Recommendations</h3>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <th>Symbol</th>
                    <th>Recommendation</th>
                    <th>Confidence</th>
                    <th>Target Weight</th>
                    <th>Target Shares</th>
                    <th>Current Shares</th>
                    <th>Gradual Entry</th>
                    <th>Current Price</th>
                </tr>
        """
        
        # Add position sizing recommendations with improved logic
        for idx, row in results_df.iterrows():
            buy_score = row['CurrentBuyScore'] if pd.notna(row['CurrentBuyScore']) else 50
            sell_score = row['CurrentSellScore'] if pd.notna(row['CurrentSellScore']) else 50
            
            # Use signal spread instead of independent scores
            signal_strength = buy_score - sell_score
            
            # Calculate confidence based on signal strength (0-1)
            confidence = min(abs(signal_strength) / 40, 1.0)
            
            # Determine recommendation based on signal spread
            if signal_strength > 30:
                rec = "STRONG BUY"
            elif signal_strength > 15:
                rec = "MODERATE BUY"
            elif signal_strength < -30:
                rec = "STRONG SELL"
            elif signal_strength < -15:
                rec = "MODERATE SELL"
            else:
                rec = "HOLD"
            
            # Scale position size gradually based on signal strength
            # 15 → ~17%, 20 → ~33%, 30 → ~50%, 40 → ~67%, 50+ → 100%
            if signal_strength > 0:
                raw_weight = min((signal_strength / 50) * 100, 100)
            elif signal_strength < 0:
                raw_weight = min((abs(signal_strength) / 50) * 100, 100)
            else:
                raw_weight = 0
            
            # Apply portfolio weight caps
            if rec == "STRONG BUY":
                target_weight = min(raw_weight * 0.1, 10)  # Max 10%
            elif rec == "MODERATE BUY":
                target_weight = min(raw_weight * 0.05, 5)  # Max 5%
            elif rec == "STRONG SELL":
                target_weight = min(raw_weight * 0.1, 10)  # Max 10% reduction
            elif rec == "MODERATE SELL":
                target_weight = min(raw_weight * 0.05, 5)  # Max 5% reduction
            else:
                target_weight = 0
            
            # Gradual entry/exit (25% increments)
            gradual_entry = "25% increments"
            
            # Get current price (approximate from market value and shares)
            if row['Shares'] > 0 and row['MarketValue'] > 0:
                current_price = row['MarketValue'] / row['Shares']
            else:
                current_price = 0
            
            # Calculate target shares based on portfolio value and target weight
            total_portfolio_value = results_df['MarketValue'].sum()
            if total_portfolio_value > 0 and current_price > 0:
                target_value = total_portfolio_value * (target_weight / 100)
                target_shares = int(target_value / current_price)
            else:
                target_shares = 0
            
            # Get current shares
            current_shares = int(row['Shares']) if pd.notna(row['Shares']) else 0
            
            # Adjust recommendation based on current position vs target
            # If already at or above target, downgrade recommendation
            if current_shares >= target_shares and rec in ["STRONG BUY", "MODERATE BUY"]:
                if rec == "STRONG BUY":
                    rec = "MODERATE BUY"
                elif rec == "MODERATE BUY":
                    rec = "HOLD"
            # If significantly below target, upgrade recommendation
            elif current_shares < target_shares * 0.5 and rec == "HOLD":
                rec = "MODERATE BUY"
            elif current_shares < target_shares * 0.25 and rec in ["HOLD", "MODERATE BUY"]:
                rec = "STRONG BUY"
            
            body += f"""
                <tr>
                    <td>{row['Symbol']}</td>
                    <td>{rec}</td>
                    <td>{confidence:.2f}</td>
                    <td>{target_weight:.1f}%</td>
                    <td>{target_shares}</td>
                    <td>{current_shares}</td>
                    <td>{gradual_entry}</td>
                    <td>${current_price:.2f}</td>
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
                <li>Position Sizing: Based on signal spread (buy_score - sell_score) with portfolio weight caps</li>
                <li>Confidence: Measures strength of buy/sell signal disagreement (0-1)</li>
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
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        
        print("\n✅ Email sent successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error sending email: {e}")
        import traceback
        traceback.print_exc()
        return False
