import streamlit as st
import boto3
import json
import os
from functools import lru_cache
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

AGENT_IMAGES = {
    "Astra": "images/Astra_icon.webp",
    "Brimstone": "images/Brimstone_icon.webp",
    "Clove": "images/Clove_icon.webp",
    "Harbor": "images/Harbor_icon.webp",
    "Omen": "images/Omen_icon.webp",
    "Viper": "images/Viper_icon.webp",
    "Iso": "images/Iso_icon.webp",
    "Jett": "images/Jett_icon.webp",
    "Neon": "images/Neon_icon.webp",
    "Phoenix": "images/Phoenix_icon.webp",
    "Raze": "images/Raze_icon.webp",
    "Reyna": "images/Reyna_icon.webp",
    "Yoru": "images/Yoru_icon.webp",
    "Breach": "images/Breach_icon.webp",
    "Fade": "images/Fade_icon.webp",
    "KAYO": "images/KAYO_icon.webp",
    "Skye": "images/Skye_icon.webp",
    "Sova": "images/Sova_icon.webp",
    "Chamber": "images/Chamber_icon.webp",
    "Cypher": "images/Cypher_icon.webp",
    "Killjoy": "images/Killjoy_icon.webp",
    "Sage": "images/Sage_icon.webp",
    "Vyse": "images/Vyse_icon.webp",
    "Gekko": "images/Gekko_icon.webp",
    "Deadlock": "images/Deadlock_icon.webp"
}

prompt_templates = {
    "professional": """Build a team using only players from VCT International. Assign roles to each player and explain why this composition would be effective in a competitive match.
    Requirements:
    1. Select 5 players covering all roles (Duelist, Controller, Initiator, Sentinel, Flex)
    2. Assign an IGL
    3. Consider agent pools and synergies
    4. Evaluate recent performances within the last year
    
    For each player provide:
    1. Role and agent recommendations (be specific with agent names)
    2. Statistical justification""",
    
    "semi_pro": """Build a team using only players from VCT Challengers. Assign roles to each player and explain why this composition would be effective in a competitive match.
    Requirements:
    1. Select 5 players covering all roles (Duelist, Controller, Initiator, Sentinel, Flex)
    2. Assign an IGL
    3. Consider agent pools and synergies
    4. Evaluate recent performances within the last year
    
    For each player provide:
    1. Role and agent recommendations (be specific with agent names)
    2. Statistical justification""",
    
    "game_changers": """Build a team using only players from VCT Game Changers. Assign roles to each player and explain why this composition would be effective in a competitive match.
    Requirements:
    1. Select 5 players covering all roles (Duelist, Controller, Initiator, Sentinel, Flex)
    2. Assign an IGL
    3. Consider agent pools and synergies
    4. Evaluate recent performances within the last year
    
    For each player provide:
    1. Role and agent recommendations (be specific with agent names)
    2. Statistical justification"""
}


load_dotenv()

# Initialize Bedrock client using environment variables
bedrock_runtime_client = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

@st.cache_data
def load_player_data():
    """Cache the player data loading"""
    with open('player_stats_ENHANCED.json', 'r') as f:
        return json.load(f)
    
    
def calculate_team_map_stats(team_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate actual average winrates for each map across team members"""
    map_stats = {}
    valid_maps = ['bind', 'split', 'haven', 'ascent', 'icebox', 'pearl', 'fracture', 'sunset', 'lotus', 'abyss', 'breeze']
    
    for map_name in valid_maps:
        winrate_key = f"{map_name}_winrate"
        # winrates for players who have data for this map
        valid_rates = [
            player.get("statistics", {}).get(winrate_key, 0)
            for player in team_data
            if player.get("statistics", {}).get(winrate_key) is not None
        ]
        
        if valid_rates:  # Only include maps where we have data
            avg_winrate = sum(valid_rates) / len(valid_rates)
            map_stats[map_name] = avg_winrate
    
    return map_stats
    
def handle_custom_query(data: Dict[str, Any], custom_query: str, player_limit: int) -> None:
    """Handle custom queries with raw LLM output display using Claude 3.5"""
    
    filtered_context = filter_context(data, "all", player_limit)
    players_info = filtered_context["players"]
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""You are a VCT expert analyst. Create a competitive 5-player team composition using ONLY players from the provided list.
You MUST follow the exact format and spacing specified below.

Available players (Top {player_limit} performers):
{json.dumps(players_info[:player_limit], indent=2)}

STRICT REQUIREMENTS:
1. MUST include EXACTLY:
   - 1 Controller (Primary agents: Brimstone, Viper, Omen, Astra, Harbor, Clove)
   - 1 Duelist (Primary agents: Phoenix, Jett, Raze, Reyna, Yoru, Neon, Iso)
   - 1 Sentinel (Primary agents: Killjoy, Cypher, Sage, Chamber, Deadlock, Vyse)
   - 1 Initiator (Primary agents: Sova, Breach, Skye, KAYO, Fade, Gekko)
   - 1 Flex (Can be either a Duelist, Sentinel, Initiator, or Controller)
2. Each player must be unique, the same player cannot be chosen twice
3. Only ONE player should be marked as IGL. This should be one of the controller players.
4. NEVER CHOOSE MORE THAN TWO (2) PLAYERS FROM THE SAME TEAM (e.g., no more than two FNATIC players per team).

CUSTOM QUERY REQUIREMENTS:
{custom_query}

FORMAT REQUIREMENTS (FOLLOW EXACTLY):

**PLAYER: [NAME]**
Current Team: [Team]
Role: [Role]
Primary Agents: [Agents list]
Backup Agents: [Agents list or None]
KDA: [KDA Ratio]
Winrate: [Overall winrate]%
Best Maps: [Top 2-3 maps with highest winrates]
Reasoning: [2 sentences including performance and map-specific strengths. If IGL, mention it here; do not mention the acronym "IGL" in the reasoning for any non-IGL player. All players MUST be referred to by either their handle or gender neutral terms (they/them/theirs).]

[Leave exactly one blank line between players]

**PLAYER: [NEXT NAME]**
[Continue exact same format for each player]

Team Analysis:
[1 sentence about team composition and synergy]
[1 sentence about strongest maps]
[1 sentence about potential weaknesses]"""
                }
            ]
        }
    ]

    try:
        response = bedrock_runtime_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "messages": messages,
                "temperature": 0.3,
                "top_p": 0.9
            }),
            contentType='application/json'
        )
        
        response_body = json.loads(response['body'].read().decode())
        response_text = response_body.get('content', [{}])[0].get('text', '')
        
        if response_text:
            tab1, tab2 = st.tabs(["Pretty View", "Raw Response"])
            
            with tab1:
                st.markdown("### Team Composition")
                display_team_composition(response_text)
                
            with tab2:
                st.markdown("### Raw LLM Response")
                st.markdown("---")
                st.markdown(response_text)
            
    except Exception as e:
        st.error(f"Error processing custom query: {str(e)}")
        st.error("If you're seeing an input length error, try being more specific in your query to reduce the data needed.")
    
def query_bedrock(prompt_type: str, context: Dict[str, Any], player_limit: int) -> str:
    """Query Amazon Bedrock with enhanced player data using Claude 3.5"""
    
    filtered_context = filter_context(context, prompt_type, player_limit)
    
    # player info format with detailed map statistics
    players_info = []
    for player in filtered_context["players"]:
        # all map winrates first
        map_winrates = {}
        for map_name in ["bind", "split", "haven", "ascent", "icebox", "pearl", "fracture", "sunset", "lotus", "breeze"]:
            winrate_key = f"{map_name}_winrate"
            if winrate := player.get("statistics", {}).get(winrate_key):
                map_winrates[map_name] = winrate
        
        # sort maps by winrate to get top performing maps
        sorted_maps = sorted(map_winrates.items(), key=lambda x: x[1], reverse=True)
        top_maps = sorted_maps[:3]  # Get top 3 maps
        
        # format map stats
        best_maps = [f"{map_name} ({winrate:.2f}%)" for map_name, winrate in top_maps]
        
        player_info = {
            "name": player["name"],
            "team": player["team"],
            "role": player["primary_role"],
            "primary_agent": player["agents"][0] if player["agents"] else "",
            "backup_agents": player["agents"][1:] if len(player["agents"]) > 1 else [],
            "kda": player["kda"],
            "region": player["region"],
            "statistics": {
                "overall_winrate": player.get("statistics", {}).get("overall_winrate", 0),
                "total_matches": player.get("statistics", {}).get("total_matches", 0),
                "best_maps": best_maps,
                "map_winrates": map_winrates
            }
        }
        players_info.append(player_info)

    # sort and limit players
    players_info.sort(key=lambda x: x["kda"], reverse=True)
    limited_players = players_info[:player_limit]
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""You are a VCT expert analyst. Create a competitive 5-player team composition using ONLY players from the provided list.
You MUST follow the exact format and spacing specified below.

Available players (Top {player_limit} performers):
{json.dumps(limited_players, indent=2)}

STRICT REQUIREMENTS:
1. MUST include EXACTLY:
   - 1 Controller (Primary agents: Brimstone, Viper, Omen, Astra, Harbor, Clove)
   - 1 Duelist (Primary agents: Phoenix, Jett, Raze, Reyna, Yoru, Neon, Iso)
   - 1 Sentinel (Primary agents: Killjoy, Cypher, Sage, Chamber, Deadlock, Vyse)
   - 1 Initiator (Primary agents: Sova, Breach, Skye, KAYO, Fade, Gekko)
   - 1 Flex (Can be either a Duelist, Sentinel, Initiator, or Controller)
2. Each player must be unique, the same player cannot be chosen twice
3. Only ONE player should be marked as IGL. This should be one of the controller players.
4. NEVER CHOOSE MORE THAN TWO (2) PLAYERS FROM THE SAME TEAM (e.g., no more than two FNATIC players per team).

FORMAT REQUIREMENTS (FOLLOW EXACTLY):

**PLAYER: [NAME]**
Current Team: [Team]
Role: [Role]
Primary Agents: [Agents list]
Backup Agents: [Agents list or None]
KDA: [KDA Ratio]
Winrate: [Overall winrate]%
Best Maps: [List exactly 3 best maps with winrates in parentheses]
Reasoning: [2 sentences including performance and map-specific strengths. If IGL, mention it here; do not mention the acronym "IGL" in the reasoning for any non-IGL player. All players MUST be referred to by either their handle or gender neutral terms (they/them/theirs).]

[Leave exactly one blank line between players]

**PLAYER: [NEXT NAME]**
[Continue exact same format for each player]

Team Analysis:
[1 sentence about team composition and synergy]
[1 sentence about strongest maps based on the overlap in players' best performing maps]
[1 sentence about potential weaknesses]"""
                }
            ]
        }
    ]

    try:
        response = bedrock_runtime_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "messages": messages,
                "temperature": 0.3,
                "top_p": 0.9
            }),
            contentType='application/json'
        )
        
        response_body = json.loads(response['body'].read().decode())
        return response_body.get('content', [{}])[0].get('text', '')
            
    except Exception as e:
        st.error(f"Error querying Bedrock: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None
    
def validate_agent_roles(agent: str, role: str) -> bool:
    """Validate that an agent matches the specified role"""
    role_agents = {
        'Duelist': ['Phoenix', 'Jett', 'Raze', 'Reyna', 'Yoru', 'Neon', 'Iso'],
        'Controller': ['Brimstone', 'Viper', 'Omen', 'Astra', 'Harbor', 'Clove'],
        'Sentinel': ['Killjoy', 'Cypher', 'Sage', 'Chamber', 'Deadlock', 'Vyse'],
        'Initiator': ['Sova', 'Breach', 'Skye', 'KAYO', 'Fade', 'Gekko'],
    }
    return agent in role_agents.get(role, [])

def validate_team_composition(team_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate team composition and agent-role consistency"""
    for player in team_data:
        role = player['role']
        agents = player['agents']
        
        # validate agent matches the role
        invalid_agents = [agent for agent in agents if not validate_agent_roles(agent, role)]
        if invalid_agents:
            st.error(f"Invalid agents {invalid_agents} for role {role} for player {player['name']}")
            return []
    
    # count IGLs
    igl_count = sum(1 for player in team_data if player['igl'])
    if igl_count != 1:
        st.error(f"Team must have exactly one IGL (currently has {igl_count})")
        return []
    
    return team_data


def filter_context(context: Dict[str, Any], prompt_type: str, player_limit: int) -> Dict[str, Any]:
    """Filter context based on prompt type with new JSON structure"""
    
    players_data = context.get("players", {})
    
    st.sidebar.write("Input context player count:", len(players_data))
    
    filtered_players = []
    
    if isinstance(players_data, dict):
        category_players = []
        for player_id, player_data in players_data.items():
            try:
                if prompt_type == "all" or \
                   (prompt_type == "professional" and player_data["team"]["category"] == "international") or \
                   (prompt_type == "semi_pro" and player_data["team"]["category"] == "challengers") or \
                   (prompt_type == "game_changers" and player_data["team"]["category"] == "game-changers") or \
                   prompt_type in ["mixed_gender", "cross_regional", "rising_star"]:
                    
                    player_info = {
                        "name": player_data["handle"],
                        "team": player_data["team"]["name"],
                        "team_category": player_data["team"]["category"],
                        "region": player_data["team"]["region"],
                        "primary_role": player_data["statistics"]["primary_role"],
                        "agents": player_data["statistics"]["most_played_agents"],
                        "kda": player_data["statistics"]["overall_kda"],
                        "statistics": {
                            "overall_winrate": player_data["statistics"]["overall_winrate"],
                            "total_matches": player_data["statistics"]["total_matches"],
                            "map_winrates": {
                                k.replace("_winrate", ""): v
                                for k, v in player_data["statistics"].items()
                                if k.endswith("_winrate") and k != "overall_winrate"
                            }
                        }
                    }
                    category_players.append(player_info)
                    
            except Exception as e:
                st.sidebar.write(f"Error processing player {player_id}: {str(e)}")
                continue
        
        filtered_players = sorted(
            category_players,
            key=lambda x: x["kda"],
            reverse=True
        )[:player_limit]

    st.sidebar.write(f"Filtered to top {len(filtered_players)} players")
    return {"players": filtered_players}

def parse_team_response(response: str) -> List[Dict[str, Any]]:
    """Parse the LLM response with enhanced format handling including team information"""
    team_data = []
    team_analysis = ""
    
    parts = response.split("Team Analysis:")
    player_section = parts[0]
    if len(parts) > 1:
        team_analysis = parts[-1].strip()
    
    player_blocks = player_section.split("**PLAYER:")
    
    player_blocks = [block for block in player_blocks if "Role:" in block]
    
    for block in player_blocks:
        player_data = {
            'name': '',
            'team': '', 
            'role': '',
            'primary_agents': [],
            'backup_agents': [],
            'kda': 0.0,
            'winrate': 0.0,
            'best_maps': [],
            'igl': False,
            'reasoning': ''
        }
        
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        if lines:
            player_data['name'] = lines[0].replace('**', '').strip()
        
        for line in lines:
            if line.startswith('Current Team:'):
                player_data['team'] = line.replace('Current Team:', '').strip()
            elif line.startswith('Role:'):
                player_data['role'] = line.replace('Role:', '').strip()
            elif line.startswith('Primary Agent:') or line.startswith('Primary Agents:'):
                agents = line.split(':', 1)[1].strip()
                player_data['primary_agents'] = [
                    a.strip() for a in agents.replace('and', ',').split(',')
                    if a.strip() and a.strip().lower() != 'none'
                ]
            elif line.startswith('Backup Agent:') or line.startswith('Backup Agents:'):
                agents = line.split(':', 1)[1].strip()
                player_data['backup_agents'] = [
                    a.strip() for a in agents.replace('and', ',').split(',')
                    if a.strip() and a.strip().lower() != 'none'
                ]
            elif line.startswith('KDA:'):
                try:
                    kda_str = line.replace('KDA:', '').strip()
                    player_data['kda'] = float(kda_str)
                except ValueError:
                    pass
            elif line.startswith('Winrate:'):
                try:
                    winrate_str = line.replace('Winrate:', '').replace('%', '').strip()
                    player_data['winrate'] = float(winrate_str)
                except ValueError:
                    pass
            elif line.startswith('Best Maps:'):
                maps_str = line.replace('Best Maps:', '').strip()
                maps = []
                for map_entry in maps_str.split(','):
                    map_name = map_entry.split('(')[0].strip()
                    if map_name:
                        maps.append(map_name)
                player_data['best_maps'] = maps
            elif line.startswith('Reasoning:'):
                player_data['reasoning'] = line.replace('Reasoning:', '').strip()
                if 'IGL' in line.upper():
                    player_data['igl'] = True
        
        if player_data['name']:  
            team_data.append(player_data)
    
    if team_analysis:
        st.session_state.team_analysis = team_analysis
        
        try:
            map_stats = {}
            for player in team_data:
                for map_name in player.get('best_maps', []):
                    map_name = map_name.lower().strip()
                    map_stats[map_name] = map_stats.get(map_name, 0) + 1
            
            team_size = len(team_data)
            if team_size > 0:
                map_stats = {k: (v / team_size) * 100 for k, v in map_stats.items()}
                st.session_state.map_stats = map_stats
        except Exception as e:
            st.sidebar.error(f"Error calculating map stats: {str(e)}")
    
    return team_data
    

def normalize_agent_name(agent: str) -> str:
    """Normalize agent name to match image filename format"""
    if agent.upper() == "KAYO" or agent.upper() == "KAY/O":
        return "KAYO"
    return agent.strip().replace("/", "").replace(" ", "")

def display_team_composition(response: str):
    """Display team composition with enhanced styling and map images"""
    try:
        team_data = parse_team_response(response)
        
        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
        
        st.markdown("<style>.player-column { margin: 0 15px; }</style>", unsafe_allow_html=True)
        cols = st.columns([1, 1, 1, 1, 1])
        
        for idx, player in enumerate(team_data):
            with cols[idx]:
                with st.container():
                    st.markdown("<div class='player-column'>", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div style='text-align: center; margin-bottom: 20px;'>
                            <h2 style='margin: 0 0 15px 0; font-size: 26px; font-weight: bold;'>{player['name']}</h2>
                            <h3 style='margin: 10px 0; font-size: 20px; color: #888;'>{player['role']}</h3>
                            <p style='margin: 5px 0; color: #666;'>{player['team']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if player['igl']:
                        st.markdown("<p style='text-align: center; margin: 5px 0;'>👑 <strong>IGL</strong></p>", unsafe_allow_html=True)
                    
                    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                    
                    if player['primary_agents']:
                        st.markdown("<p style='text-align: center; font-weight: bold; margin: 10px 0;'>Primary Agents</p>", unsafe_allow_html=True)
                        
                        agents_count = len(player['primary_agents'])
                        if agents_count > 2:
                            grid_cols = 2
                            rows = (agents_count + 1) // 2
                        else:
                            # for 2 agents use single column
                            grid_cols = agents_count
                            rows = 1
                        
                        for row in range(rows):
                            agent_cols = st.columns(grid_cols)
                            for col in range(grid_cols):
                                agent_idx = row * grid_cols + col
                                if agent_idx < agents_count:
                                    agent = player['primary_agents'][agent_idx]
                                    with agent_cols[col]:
                                        try:
                                            normalized_agent = normalize_agent_name(agent)
                                            image_path = f"images/{normalized_agent}_icon.webp"
                                            
                                            if Path(image_path).exists():
                                                # Adjust image size based on grid
                                                img_width = 60 if agents_count > 2 else 80
                                                st.image(image_path, width=img_width, caption=agent, use_column_width=False)
                                            else:
                                                st.markdown(f"<p style='text-align: center;'>{agent}</p>", unsafe_allow_html=True)
                                        except Exception as e:
                                            st.sidebar.error(f"Error loading {agent}: {str(e)}")
                                            st.markdown(f"<p style='text-align: center;'>{agent}</p>", unsafe_allow_html=True)
                    
                    if player['backup_agents']:
                        st.markdown("<p style='text-align: center; font-weight: bold; margin: 10px 0;'>Backup Agents</p>", unsafe_allow_html=True)
                        backup_count = len(player['backup_agents'])
                        backup_cols = st.columns(min(backup_count, 3))
                        
                        for i, agent in enumerate(player['backup_agents']):
                            with backup_cols[i % 3]:
                                try:
                                    normalized_agent = normalize_agent_name(agent)
                                    image_path = f"images/{normalized_agent}_icon.webp"
                                    
                                    if Path(image_path).exists():
                                        st.image(image_path, width=40, caption=agent, use_column_width=False)
                                    else:
                                        st.markdown(f"<p style='text-align: center;'>{agent}</p>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.sidebar.error(f"Error loading backup {agent}: {str(e)}")
                                    st.markdown(f"<p style='text-align: center;'>{agent}</p>", unsafe_allow_html=True)
                    
                    st.markdown(f"<p style='text-align: center; margin: 10px 0;'><strong>KDA:</strong> {player['kda']:.2f}</p>", unsafe_allow_html=True)
                    if 'winrate' in player:
                        st.markdown(f"<p style='text-align: center; margin: 10px 0;'><strong>Winrate:</strong> {player['winrate']:.1f}%</p>", unsafe_allow_html=True)
                    
                    with st.expander("Analysis"):
                        st.markdown("""
                            <div style='margin: -1rem -1.5rem;'>
                                <div style='padding: 1rem 1.5rem;'>
                        """, unsafe_allow_html=True)
                        st.markdown(f"<p style='margin: 0;'>{player['reasoning']}</p>", unsafe_allow_html=True)
                        st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'team_analysis'):
            st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
            st.markdown("""
                <h2 style='text-align: center; margin: 20px 0;'>Team Analysis</h2>
            """, unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: justify;'>{st.session_state.team_analysis}</p>", unsafe_allow_html=True)
            
            if hasattr(st.session_state, 'map_stats'):
                st.markdown("### Winrate of Top 3 Team Maps")
                
                map_stats = st.session_state.map_stats
                sorted_maps = sorted(map_stats.items(), key=lambda x: x[1], reverse=True)[:3]  # Limit to top 3
                
                # proceed if we have maps to display
                if sorted_maps:
                    cols = st.columns(3)
                    for idx, (map_name, winrate) in enumerate(sorted_maps):
                        if idx < len(cols):  # don't exceed column count
                            with cols[idx]:
                                try:
                                    image_path = f"images/Loading_Screen_{map_name.title()}.webp"
                                    if Path(image_path).exists():
                                        st.image(image_path, use_column_width=True)
                                    st.metric(
                                        label=map_name.title(),
                                        value=f"{winrate:.1f}%",
                                        delta=None
                                    )
                                except Exception as e:
                                    st.sidebar.error(f"Error loading map {map_name}: {str(e)}")
                                    continue
            
    except Exception as e:
        st.error(f"Error displaying team composition: {str(e)}")
        st.text("Raw response:")
        st.text(response)

def main():
    st.title("VCT Team Builder Digital Assistant")
    
    #player limit selector in sidebar
    st.sidebar.header("Query Settings")
    player_limit = st.sidebar.slider(
        "Number of players to consider",
        min_value=200,
        max_value=1700,
        value=200,
        step=10,
        help="Higher values will include more players but increase response time"
    )
    
    # load data once and cache it
    try:
        data = load_player_data()
    except FileNotFoundError:
        st.error("Player data file not found. Please ensure player_stats_ENHANCED.json exists in the current directory.")
        return

    st.sidebar.header("Team Building Options")
    team_type = st.sidebar.selectbox(
        "Select Team Type",
        [
            "Professional (VCT International)",
            "Semi-Professional (VCT Challengers)",
            "Game Changers (VCT Game Changers)"
        ]
    )
    
    prompt_type_mapping = {
        "Professional (VCT International)": "professional",
        "Semi-Professional (VCT Challengers)": "semi_pro",
        "Game Changers (VCT Game Changers)": "game_changers"
    }
    
    if st.button("Generate Team"):
        with st.spinner("Analyzing players and building team..."):
            response = query_bedrock(prompt_type_mapping[team_type], data, player_limit)
            
            if response:
                display_team_composition(response)
    
    st.sidebar.header("Options")
    analysis_type = st.sidebar.selectbox(
        "Select Type",
        [
            "Click to Generate",
            "Custom Query"
        ]
    )
    
    if analysis_type == "Custom Query":
        st.markdown("""
        ### Custom Query
        """)
        
        custom_query = st.text_area(
            "Enter your Custom Query:",
            height=100,
            help="Query Box"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("Generate Team (Custom Query)", key="custom_query_button")
        with col2:
            if not custom_query.strip() and analyze_button:
                st.warning("Please enter a query before analyzing.")
        
        if analyze_button and custom_query.strip():
            with st.spinner("Analyzing and building team..."):
                try:
                    handle_custom_query(data, custom_query, player_limit)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Try adjusting the player limit or being more specific in your query.")

if __name__ == "__main__":
    main()