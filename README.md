# VCT Team Builder Assistant

A team composition analysis tool built for the DEVPOST VCT Hackathon. This application uses AWS Bedrock (Claude 3.5 Sonnet) and Streamlit to help build optimal VALORANT team compositions using real player data.

## Features

- Multiple team building options:
  - Professional (VCT International)
  - Semi-Professional (VCT Challengers)
  - Game Changers (VCT Game Changers)
  - Custom queries for more robust team options
- Visual team compositions with agent icons
- Map performance analysis
- Detailed player statistics
- IGL designation and analysis

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vct-team-builder.git
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Set up AWS credentials in a `.env` file:
```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=your_region
```

4. Run the application:
```bash
streamlit run streamlit_app.py
```

## Stack

- AWS Bedrock (and custom knowledge base) with Claude 3.5 Sonnet
- Python + Streamlit
- VCT match + player data parsed into a comprehensive json file

## Credits

- AWS and Riot Games for the hackathon opportunity
- Riot Games for the VCT game + player data


*This is was a hackathon project, and my first time ever using an LLM or Amazon Bedrock*