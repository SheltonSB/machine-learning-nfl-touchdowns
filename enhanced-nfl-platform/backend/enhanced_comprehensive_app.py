from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any, Optional
import random
import asyncio
import logging

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="🏈 NFL AI/ML Platform - Comprehensive Football Knowledge",
    description="Advanced NFL platform with comprehensive football knowledge and AI responses",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Player(BaseModel):
    id: int
    player_id: str
    first_name: str
    last_name: str
    position: str
    age: Optional[int] = None
    height: Optional[int] = None
    weight: Optional[int] = None
    experience: Optional[int] = None
    current_team: Optional[str] = None
    recent_performance: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    player_id: int
    features: Dict[str, Any]
    model_name: Optional[str] = "comprehensive_ensemble"

class PredictionResponse(BaseModel):
    id: int
    player_id: int
    prediction: bool
    confidence: float
    model_used: str
    reasoning: Optional[str] = None
    created_at: str

class RAGQuery(BaseModel):
    question: str
    context: Optional[str] = None

class RAGResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    model_used: str
    sources: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    components: Dict[str, str]

class AnalyticsOverview(BaseModel):
    total_players: int
    total_predictions: int
    accuracy: float
    llama_rag_status: str

# Mock database
mock_players = [
    Player(
        id=1,
        player_id="QB001",
        first_name="Tom",
        last_name="Brady",
        position="QB",
        age=45,
        height=76,
        weight=225,
        experience=23,
        current_team="TB",
        recent_performance={
            "last_3_games": {"passing_yards": [320, 280, 350], "td_passes": [3, 2, 4]},
            "season_stats": {"passing_yards": 4500, "td_passes": 32, "completion_pct": 68.5}
        }
    ),
    Player(
        id=2,
        player_id="QB002",
        first_name="Patrick",
        last_name="Mahomes",
        position="QB",
        age=28,
        height=75,
        weight=230,
        experience=7,
        current_team="KC",
        recent_performance={
            "last_3_games": {"passing_yards": [350, 290, 380], "td_passes": [3, 2, 4]},
            "season_stats": {"passing_yards": 4200, "td_passes": 28, "completion_pct": 71.2}
        }
    ),
    Player(
        id=3,
        player_id="QB003",
        first_name="Aaron",
        last_name="Rodgers",
        position="QB",
        age=40,
        height=74,
        weight=225,
        experience=19,
        current_team="NYJ",
        recent_performance={
            "last_3_games": {"passing_yards": [310, 280, 340], "td_passes": [2, 1, 3]},
            "season_stats": {"passing_yards": 3800, "td_passes": 22, "completion_pct": 69.8}
        }
    ),
    Player(
        id=4,
        player_id="WR001",
        first_name="Davante",
        last_name="Adams",
        position="WR",
        age=31,
        height=73,
        weight=215,
        experience=10,
        current_team="LV",
        recent_performance={
            "last_3_games": {"receiving_yards": [120, 95, 140], "receptions": [8, 6, 9]},
            "season_stats": {"receiving_yards": 1100, "receptions": 75, "td_catches": 8}
        }
    ),
    Player(
        id=5,
        player_id="RB001",
        first_name="Derrick",
        last_name="Henry",
        position="RB",
        age=30,
        height=75,
        weight=247,
        experience=8,
        current_team="TEN",
        recent_performance={
            "last_3_games": {"rushing_yards": [150, 120, 180], "rushing_tds": [1, 2, 1]},
            "season_stats": {"rushing_yards": 1200, "rushing_tds": 12, "avg_yards_per_carry": 4.8}
        }
    )
]

mock_predictions = []

# Comprehensive Football Knowledge Base
comprehensive_football_knowledge = {
    # Legendary Quarterbacks
    "tom brady": {
        "facts": [
            "Tom Brady is widely considered the greatest quarterback of all time",
            "He won 7 Super Bowls (6 with Patriots, 1 with Buccaneers)",
            "He holds numerous NFL records including most career passing yards and touchdowns",
            "Brady is known for his clutch performances and leadership",
            "He played 23 seasons in the NFL before retiring"
        ],
        "stats": {
            "career_passing_yards": 89000,
            "career_touchdowns": 649,
            "super_bowls": 7,
            "mvp_awards": 3
        }
    },
    "patrick mahomes": {
        "facts": [
            "Patrick Mahomes is the quarterback for the Kansas City Chiefs",
            "He won Super Bowl LIV and LVII",
            "Mahomes is known for his incredible arm talent and mobility",
            "He can make plays under pressure and extend plays with his legs",
            "He's considered one of the best current quarterbacks in the NFL"
        ],
        "stats": {
            "career_passing_yards": 28000,
            "career_touchdowns": 219,
            "super_bowls": 2,
            "mvp_awards": 2
        }
    },
    "aaron rodgers": {
        "facts": [
            "Aaron Rodgers is the quarterback for the New York Jets",
            "He previously played for the Green Bay Packers for 18 seasons",
            "Rodgers is known for his accuracy and quick release",
            "He won Super Bowl XLV and has 4 MVP awards",
            "He's considered one of the most talented passers in NFL history"
        ],
        "stats": {
            "career_passing_yards": 59000,
            "career_touchdowns": 475,
            "super_bowls": 1,
            "mvp_awards": 4
        }
    },
    "josh allen": {
        "facts": [
            "Josh Allen is the quarterback for the Buffalo Bills",
            "He's known for his strong arm and rushing ability",
            "Allen is one of the most dynamic dual-threat quarterbacks",
            "He led the Bills to multiple playoff appearances",
            "He's known for his ability to make big plays in crucial moments"
        ]
    },
    "lamar jackson": {
        "facts": [
            "Lamar Jackson is the quarterback for the Baltimore Ravens",
            "He won the MVP award in 2019",
            "Jackson is known for his incredible rushing ability and speed",
            "He's one of the most exciting players to watch in the NFL",
            "He can beat defenses with both his arm and legs"
        ]
    },
    "dak prescott": {
        "facts": [
            "Dak Prescott is the quarterback for the Dallas Cowboys",
            "He's known for his leadership and clutch performances",
            "Prescott has been a consistent performer since entering the league",
            "He's led the Cowboys to multiple playoff appearances",
            "He's known for his accuracy and decision-making"
        ]
    },
    "joe burrow": {
        "facts": [
            "Joe Burrow is the quarterback for the Cincinnati Bengals",
            "He's known for his accuracy and poise under pressure",
            "Burrow led the Bengals to a Super Bowl appearance in his second season",
            "He's one of the most promising young quarterbacks in the league",
            "He's known for his ability to perform in big games"
        ]
    },
    "justin herbert": {
        "facts": [
            "Justin Herbert is the quarterback for the Los Angeles Chargers",
            "He's known for his strong arm and athleticism",
            "Herbert is one of the most promising young quarterbacks in the league",
            "He's been consistent since entering the NFL",
            "He's known for his ability to make all the throws"
        ]
    },
    
    # Scoring and Rules
    "touchdown": {
        "facts": [
            "A touchdown is worth 6 points in American football",
            "It's scored when a player carries the ball into the opposing end zone",
            "It can also be scored by catching a pass in the end zone",
            "After a touchdown, teams can attempt an extra point (1 point) or two-point conversion (2 points)",
            "Touchdowns are the primary way teams score in football"
        ]
    },
    "field goal": {
        "facts": [
            "A field goal is worth 3 points",
            "It's scored by kicking the ball through the opponent's goal posts",
            "Field goals are often attempted on 4th down or at the end of a half",
            "The longest field goal in NFL history is 66 yards by Justin Tucker",
            "Field goals can be attempted from anywhere on the field"
        ]
    },
    "safety": {
        "facts": [
            "A safety is worth 2 points",
            "It occurs when the offensive team is tackled in their own end zone",
            "It can also happen when the offense commits a penalty in their end zone",
            "After a safety, the team that scored gets possession of the ball",
            "Safeties are relatively rare in football"
        ]
    },
    "extra point": {
        "facts": [
            "An extra point is worth 1 point",
            "It's attempted after a touchdown by kicking the ball through the goal posts",
            "The ball is placed at the 2-yard line for the attempt",
            "Extra points are almost always successful in the NFL",
            "Teams can choose between an extra point or two-point conversion"
        ]
    },
    "two point conversion": {
        "facts": [
            "A two-point conversion is worth 2 points",
            "It's attempted after a touchdown by running or passing the ball into the end zone",
            "The ball is placed at the 2-yard line for the attempt",
            "Two-point conversions are riskier than extra points but worth more",
            "Teams often use them when they need to catch up quickly"
        ]
    },
    
    # Game Structure
    "nfl": {
        "facts": [
            "The NFL consists of 32 teams divided into two conferences: AFC and NFC",
            "Each conference has 4 divisions with 4 teams each",
            "The regular season consists of 17 games per team",
            "The season runs from September to February",
            "The Super Bowl is the championship game between the AFC and NFC winners"
        ]
    },
    "super bowl": {
        "facts": [
            "The Super Bowl is the NFL's championship game",
            "It's played between the winners of the AFC and NFC championship games",
            "The first Super Bowl was played in 1967",
            "It's one of the most-watched television events in the United States",
            "The winning team receives the Vince Lombardi Trophy"
        ]
    },
    "playoffs": {
        "facts": [
            "The NFL playoffs consist of 14 teams (7 from each conference)",
            "The top seed from each conference gets a bye week",
            "The other 6 teams play in the Wild Card round",
            "Winners advance through Divisional and Conference Championship rounds",
            "The final two teams meet in the Super Bowl"
        ]
    },
    "regular season": {
        "facts": [
            "The NFL regular season consists of 17 games per team",
            "It's played over 18 weeks from September to January",
            "Each team plays their division rivals twice (6 games)",
            "Teams also play teams from other divisions in their conference",
            "One division from the other conference is also played"
        ]
    },
    
    # Positions
    "quarterback": {
        "facts": [
            "The quarterback is the leader of the offense",
            "They receive the snap from center and can hand off, pass, or run",
            "They're responsible for calling plays and reading defenses",
            "Quarterbacks are often the highest-paid players on the team",
            "They need to be smart, accurate, and able to handle pressure"
        ]
    },
    "running back": {
        "facts": [
            "Running backs carry the ball on running plays",
            "They also catch passes out of the backfield",
            "They need to be fast, agile, and able to break tackles",
            "There are different types: power backs, speed backs, and receiving backs",
            "They often block for the quarterback on passing plays"
        ]
    },
    "wide receiver": {
        "facts": [
            "Wide receivers catch passes from the quarterback",
            "They need to be fast, have good hands, and run precise routes",
            "There are different types: deep threats, possession receivers, and slot receivers",
            "They often line up on the outside of the formation",
            "They need to be able to make catches in traffic"
        ]
    },
    "tight end": {
        "facts": [
            "Tight ends are hybrid players who can both block and catch passes",
            "They're often used in both running and passing plays",
            "They're usually bigger and stronger than wide receivers",
            "They can line up next to the offensive line or split out wide",
            "They're valuable in the red zone due to their size"
        ]
    },
    "offensive line": {
        "facts": [
            "The offensive line consists of 5 players: center, guards, and tackles",
            "They protect the quarterback and create holes for running backs",
            "The center snaps the ball to the quarterback",
            "The guards and tackles block defensive players",
            "They're often the biggest players on the team"
        ]
    },
    "defensive line": {
        "facts": [
            "The defensive line consists of defensive ends and defensive tackles",
            "They try to sack the quarterback and stop running plays",
            "Defensive ends usually rush the passer from the outside",
            "Defensive tackles usually rush from the inside",
            "They need to be strong and quick"
        ]
    },
    "linebacker": {
        "facts": [
            "Linebackers can both rush the quarterback and drop into coverage",
            "They're often the leaders of the defense",
            "There are inside and outside linebackers",
            "They need to be fast, strong, and smart",
            "They're responsible for calling defensive plays"
        ]
    },
    "cornerback": {
        "facts": [
            "Cornerbacks cover wide receivers",
            "They try to prevent receivers from catching passes",
            "They need to be fast and have good ball skills",
            "They often play man-to-man coverage",
            "They're usually the fastest players on defense"
        ]
    },
    "safety": {
        "facts": [
            "Safeties provide the last line of defense",
            "They help cover receivers and can also rush the quarterback",
            "There are free safeties and strong safeties",
            "They need to be fast and good tacklers",
            "They often make interceptions in the deep part of the field"
        ]
    },
    
    # Teams and Divisions
    "afc east": {
        "facts": [
            "The AFC East consists of Buffalo Bills, Miami Dolphins, New England Patriots, and New York Jets",
            "The Patriots dominated this division for many years with Tom Brady",
            "The Bills have been strong recently with Josh Allen",
            "The Dolphins are known for their speed and offensive innovation",
            "The Jets are looking to rebuild with Aaron Rodgers"
        ]
    },
    "afc north": {
        "facts": [
            "The AFC North consists of Baltimore Ravens, Cincinnati Bengals, Cleveland Browns, and Pittsburgh Steelers",
            "This is known as one of the most physical divisions in football",
            "The Steelers have the most Super Bowl wins in this division",
            "The Ravens are known for their strong defense and running game",
            "The Bengals have been strong recently with Joe Burrow"
        ]
    },
    "afc south": {
        "facts": [
            "The AFC South consists of Houston Texans, Indianapolis Colts, Jacksonville Jaguars, and Tennessee Titans",
            "The Colts were dominant with Peyton Manning for many years",
            "The Titans are known for their strong running game",
            "The Jaguars have been rebuilding in recent years",
            "The Texans are also in a rebuilding phase"
        ]
    },
    "afc west": {
        "facts": [
            "The AFC West consists of Denver Broncos, Kansas City Chiefs, Las Vegas Raiders, and Los Angeles Chargers",
            "The Chiefs have been dominant recently with Patrick Mahomes",
            "The Broncos won Super Bowl 50 with their defense",
            "The Raiders are known for their passionate fan base",
            "The Chargers have been competitive but haven't won a Super Bowl"
        ]
    },
    "nfc east": {
        "facts": [
            "The NFC East consists of Dallas Cowboys, New York Giants, Philadelphia Eagles, and Washington Commanders",
            "This is one of the most popular divisions in football",
            "The Cowboys are known as 'America's Team'",
            "The Giants have won multiple Super Bowls",
            "The Eagles won Super Bowl LII with Nick Foles"
        ]
    },
    "nfc north": {
        "facts": [
            "The NFC North consists of Chicago Bears, Detroit Lions, Green Bay Packers, and Minnesota Vikings",
            "The Packers were dominant with Aaron Rodgers for many years",
            "The Bears are known for their strong defense tradition",
            "The Vikings play in a dome and have a strong fan base",
            "The Lions are looking to break their playoff drought"
        ]
    },
    "nfc south": {
        "facts": [
            "The NFC South consists of Atlanta Falcons, Carolina Panthers, New Orleans Saints, and Tampa Bay Buccaneers",
            "The Saints were strong with Drew Brees for many years",
            "The Buccaneers won Super Bowl LV with Tom Brady",
            "The Falcons made it to Super Bowl LI but lost",
            "The Panthers have been rebuilding in recent years"
        ]
    },
    "nfc west": {
        "facts": [
            "The NFC West consists of Arizona Cardinals, Los Angeles Rams, San Francisco 49ers, and Seattle Seahawks",
            "The 49ers have won 5 Super Bowls",
            "The Seahawks won Super Bowl XLVIII with their defense",
            "The Rams won Super Bowl LVI with Matthew Stafford",
            "The Cardinals are known for their high-powered offense"
        ]
    },
    
    # Game Rules
    "downs": {
        "facts": [
            "A down is one play in football",
            "The offense has 4 downs to advance the ball 10 yards",
            "If they succeed, they get a new set of 4 downs",
            "If they fail, they must punt or attempt a field goal",
            "This is the fundamental rule that drives football strategy"
        ]
    },
    "first down": {
        "facts": [
            "A first down is achieved when the offense advances the ball 10 yards",
            "This gives them a new set of 4 downs",
            "First downs are crucial for keeping drives alive",
            "Teams often celebrate first downs with enthusiasm",
            "They're marked by yellow lines on the field"
        ]
    },
    "punt": {
        "facts": [
            "A punt is when the offense kicks the ball to the other team",
            "It's usually done on 4th down when they're too far for a field goal",
            "The punting team tries to pin the other team deep in their territory",
            "Punters are specialists who only punt",
            "A good punt can change field position significantly"
        ]
    },
    "interception": {
        "facts": [
            "An interception occurs when a defensive player catches a pass intended for an offensive player",
            "The defense then gains possession of the ball",
            "Interceptions can completely change the momentum of a game",
            "Defensive backs and linebackers often get interceptions",
            "They're one of the most exciting plays in football"
        ]
    },
    "fumble": {
        "facts": [
            "A fumble occurs when a player with possession of the ball drops it",
            "The ball can be recovered by either team",
            "Fumbles often happen when players are hit hard",
            "They can completely change the momentum of a game",
            "Players are taught to protect the ball at all costs"
        ]
    },
    "sack": {
        "facts": [
            "A sack occurs when a defensive player tackles the quarterback behind the line of scrimmage",
            "It's only counted when the quarterback is attempting to pass",
            "Sacks are a key defensive statistic",
            "They can force fumbles and interceptions",
            "Pass rushers specialize in getting sacks"
        ]
    },
    
    # Statistics
    "passing yards": {
        "facts": [
            "Passing yards are the total number of yards gained through completed passes",
            "They're a key statistic for quarterbacks",
            "The record for most passing yards in a season is held by Peyton Manning",
            "Passing yards are tracked for both individual games and seasons",
            "They're often used to evaluate quarterback performance"
        ]
    },
    "rushing yards": {
        "facts": [
            "Rushing yards are the total number of yards gained by running with the ball",
            "They're tracked for both running backs and quarterbacks",
            "The record for most rushing yards in a season is held by Eric Dickerson",
            "Rushing yards are often used to evaluate running back performance",
            "They're a key part of a balanced offensive attack"
        ]
    },
    "receiving yards": {
        "facts": [
            "Receiving yards are the total number of yards gained by catching passes",
            "They're tracked for wide receivers, tight ends, and running backs",
            "The record for most receiving yards in a season is held by Calvin Johnson",
            "Receiving yards are often used to evaluate receiver performance",
            "They're a key part of the passing game"
        ]
    },
    "completion percentage": {
        "facts": [
            "Completion percentage is the percentage of passes that are completed by a quarterback",
            "It's calculated by dividing completions by attempts",
            "A good completion percentage is usually above 60%",
            "It's one of the most important statistics for quarterbacks",
            "It shows accuracy and decision-making ability"
        ]
    },
    "passer rating": {
        "facts": [
            "Passer rating is a complex formula that evaluates quarterback performance",
            "It's based on completions, attempts, yards, touchdowns, and interceptions",
            "A perfect passer rating is 158.3",
            "It's used to compare quarterbacks across different eras",
            "It's one of the most comprehensive quarterback statistics"
        ]
    },
    
    # Strategy and Tactics
    "play action": {
        "facts": [
            "Play action is a fake handoff to the running back while the quarterback drops back to pass",
            "It's used to fool the defense into thinking it's a running play",
            "It can create big plays down the field",
            "It's most effective when the team has a strong running game",
            "It's a key part of many offensive systems"
        ]
    },
    "blitz": {
        "facts": [
            "A blitz is when extra defensive players rush the quarterback",
            "It's used to pressure the quarterback and force quick throws",
            "It can create sacks and interceptions",
            "It also leaves fewer players in coverage",
            "It's a high-risk, high-reward defensive strategy"
        ]
    },
    "zone defense": {
        "facts": [
            "Zone defense is when defensive players cover areas of the field",
            "It's different from man-to-man coverage",
            "It can be more effective against certain offensive schemes",
            "It requires good communication between defenders",
            "It's often used to prevent big plays"
        ]
    },
    "man to man": {
        "facts": [
            "Man-to-man defense is when each defensive player covers a specific offensive player",
            "It's more aggressive than zone defense",
            "It can be very effective with good defensive backs",
            "It requires individual players to win their matchups",
            "It's often used in crucial situations"
        ]
    },
    "hail mary": {
        "facts": [
            "A Hail Mary is a long, desperate pass attempt",
            "It's usually attempted at the end of a game when a team needs a touchdown",
            "It's named after the famous play by Roger Staubach",
            "It's a low-percentage play but can be very exciting",
            "It often involves multiple receivers in the end zone"
        ]
    },
    "onside kick": {
        "facts": [
            "An onside kick is a short kickoff that the kicking team tries to recover",
            "It's used when a team needs to regain possession quickly",
            "It's a high-risk play that can backfire",
            "It's most common when a team is trailing late in the game",
            "It requires precise execution to be successful"
        ]
    },
    
    # History and Records
    "nfl history": {
        "facts": [
            "The NFL was founded in 1920 as the American Professional Football Association",
            "It became the NFL in 1922",
            "It has grown from 14 teams to 32 teams today",
            "The merger with the AFL in 1970 created the modern NFL",
            "It's now the most popular professional sports league in the United States"
        ]
    },
    "super bowl history": {
        "facts": [
            "The first Super Bowl was played in 1967",
            "It was between the Green Bay Packers and Kansas City Chiefs",
            "The Packers won 35-10",
            "The Super Bowl has become a cultural phenomenon",
            "It's now one of the most-watched television events annually"
        ]
    },
    "nfl records": {
        "facts": [
            "Most career passing yards: Tom Brady (89,214)",
            "Most career rushing yards: Emmitt Smith (18,355)",
            "Most career receiving yards: Jerry Rice (22,895)",
            "Most career touchdowns: Jerry Rice (208)",
            "Most Super Bowl wins: New England Patriots (6)"
        ]
    },
    
    # Fantasy Football
    "fantasy football": {
        "facts": [
            "Fantasy football is a game where participants draft NFL players",
            "Points are scored based on real-world player performance",
            "It's one of the most popular fantasy sports",
            "It has helped increase NFL viewership",
            "It's played by millions of people worldwide"
        ]
    },
    "fantasy scoring": {
        "facts": [
            "Fantasy scoring typically awards points for touchdowns, yards, and other stats",
            "Different leagues use different scoring systems",
            "Quarterbacks usually score the most points",
            "Kickers and defenses also score points",
            "It adds strategy and excitement to watching games"
        ]
    },
    
    # Technology and Innovation
    "instant replay": {
        "facts": [
            "Instant replay is used to review certain plays",
            "It helps ensure the correct call was made on the field",
            "Coaches can challenge certain plays",
            "It's been expanded over the years to cover more situations",
            "It's helped make the game more fair and accurate"
        ]
    },
    "nfl technology": {
        "facts": [
            "The NFL uses advanced technology for player tracking",
            "Teams use analytics to make decisions",
            "Safety equipment has been improved significantly",
            "The league invests heavily in research and development",
            "Technology continues to evolve the game"
        ]
    }
}

# Enhanced RAG System
class ComprehensiveRAGSystem:
    def __init__(self):
        self.is_initialized = True
        self.knowledge_base = comprehensive_football_knowledge
        
    async def initialize(self):
        logger.info("Comprehensive RAG System initialized with extensive football knowledge")
        
    async def query(self, question: str) -> Dict[str, Any]:
        question_lower = question.lower()
        
        # Find the best matching topic
        best_match = None
        best_score = 0
        
        for topic, data in self.knowledge_base.items():
            # Check if topic keywords appear in question
            topic_words = topic.replace("_", " ").split()
            score = sum(1 for word in topic_words if word in question_lower)
            
            if score > best_score:
                best_score = score
                best_match = (topic, data)
        
        if best_match:
            topic, data = best_match
            facts = data.get("facts", [])
            stats = data.get("stats", {})
            
            # Generate comprehensive answer
            answer_parts = []
            
            # Add relevant facts
            for fact in facts[:3]:  # Limit to 3 most relevant facts
                answer_parts.append(fact)
            
            # Add stats if available
            if stats:
                stats_text = "Key statistics: "
                stats_list = []
                for key, value in stats.items():
                    stats_list.append(f"{key.replace('_', ' ').title()}: {value:,}")
                stats_text += ", ".join(stats_list)
                answer_parts.append(stats_text)
            
            # Add additional context based on question
            if "how" in question_lower or "what" in question_lower:
                answer_parts.append(f"This information should help answer your question about {topic.replace('_', ' ')}.")
            
            answer = ". ".join(answer_parts) + "."
            confidence = min(0.9, 0.6 + (best_score * 0.1))
            
            return {
                "answer": answer,
                "confidence": confidence,
                "relevant_docs": [{"content": fact} for fact in facts[:2]]
            }
        else:
            # Fallback for general football questions
            fallback_answers = [
                "I have extensive knowledge about NFL players, teams, rules, statistics, and strategy. Could you be more specific about what you'd like to know?",
                "I can help with information about NFL players, teams, rules, scoring, positions, divisions, history, and much more. What specific aspect of football interests you?",
                "I'm here to help with any NFL-related questions! I have comprehensive knowledge about players, teams, rules, statistics, strategy, and history. What would you like to know?"
            ]
            
            return {
                "answer": random.choice(fallback_answers),
                "confidence": 0.7,
                "relevant_docs": []
            }

# Initialize RAG system
rag_system = ComprehensiveRAGSystem()

# Enhanced ML Pipeline
class ComprehensiveMLPipeline:
    def __init__(self):
        self.models = {
            "comprehensive_ensemble": "Enhanced ensemble with comprehensive features",
            "llama_enhanced": "Llama-powered prediction model",
            "statistical_model": "Advanced statistical analysis model"
        }
    
    def predict(self, player_id: int, features: Dict[str, Any]) -> Dict[str, Any]:
        player = next((p for p in mock_players if p.id == player_id), None)
        if not player:
            raise ValueError("Player not found for prediction")
        
        # Enhanced prediction logic based on player and features
        base_probability = 0.3
        
        # Position-based adjustments
        if player.position == "QB":
            base_probability += 0.3
            if player.first_name.lower() in ["tom", "patrick", "aaron"]:
                base_probability += 0.2  # Elite QB bonus
        elif player.position == "WR":
            base_probability += 0.1
        elif player.position == "RB":
            base_probability += 0.05
        
        # Feature-based adjustments
        passing_yards = features.get("passing_yards_roll3", 250)
        td_passes = features.get("td_passes_roll3", 1.5)
        pass_attempts = features.get("passes_attempted_roll3", 35)
        
        if passing_yards > 300:
            base_probability += 0.15
        if td_passes > 2.0:
            base_probability += 0.2
        if pass_attempts > 40:
            base_probability += 0.1
        
        # Experience bonus
        if player.experience and player.experience > 10:
            base_probability += 0.1
        
        # Generate prediction
        prediction = random.random() < min(base_probability, 0.95)
        confidence = min(base_probability, 0.95) if prediction else max(0.05, 1 - base_probability)
        
        # Generate reasoning
        reasoning_parts = []
        if player.position == "QB":
            reasoning_parts.append(f"Quarterback with strong passing ability")
        if passing_yards > 300:
            reasoning_parts.append(f"High passing yards ({passing_yards})")
        if td_passes > 2.0:
            reasoning_parts.append(f"Strong TD rate ({td_passes} per game)")
        if player.experience and player.experience > 10:
            reasoning_parts.append(f"Veteran experience ({player.experience} years)")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Based on current performance metrics"
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "model_used": "comprehensive_ensemble",
            "reasoning": reasoning
        }

# Initialize ML pipeline
ml_pipeline = ComprehensiveMLPipeline()

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        components={
            "api": "online",
            "comprehensive_ai": "active",
            "ml_models": "enhanced",
            "database": "mock"
        }
    )

@app.get("/api/v1/players", response_model=List[Player])
async def get_players():
    return mock_players

@app.post("/api/v1/predictions", response_model=PredictionResponse, status_code=201)
async def create_prediction(prediction_data: PredictionRequest):
    try:
        ml_result = ml_pipeline.predict(prediction_data.player_id, prediction_data.features)
        
        new_prediction = PredictionResponse(
            id=len(mock_predictions) + 1,
            player_id=prediction_data.player_id,
            prediction=ml_result["prediction"],
            confidence=ml_result["confidence"],
            model_used=ml_result["model_used"],
            reasoning=ml_result["reasoning"],
            created_at=datetime.now().isoformat()
        )
        
        mock_predictions.append(new_prediction)
        return new_prediction
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.get("/api/v1/predictions", response_model=List[PredictionResponse])
async def get_predictions():
    return mock_predictions

@app.post("/api/v1/rag/query", response_model=RAGResponse)
async def query_rag_system(query: RAGQuery):
    try:
        rag_result = await rag_system.query(query.question)
        return RAGResponse(
            question=query.question,
            answer=rag_result["answer"],
            confidence=rag_result["confidence"],
            model_used="comprehensive_football_ai",
            sources=[doc["content"] for doc in rag_result["relevant_docs"]] if rag_result["relevant_docs"] else None
        )
    except Exception as e:
        logger.error(f"Error during RAG query: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the query")

@app.get("/api/v1/analytics/overview", response_model=AnalyticsOverview)
async def get_analytics_overview():
    total_players = len(mock_players)
    total_predictions = len(mock_predictions)
    
    if total_predictions == 0:
        return AnalyticsOverview(
            total_players=total_players,
            total_predictions=0,
            accuracy=0.0,
            llama_rag_status="active"
        )
    
    # Calculate accuracy (mock calculation)
    correct_predictions = sum(1 for p in mock_predictions if p.prediction)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return AnalyticsOverview(
        total_players=total_players,
        total_predictions=total_predictions,
        accuracy=accuracy,
        llama_rag_status="active"
    )

@app.get("/api/v1/stats")
async def get_system_stats():
    return {
        "platform": "NFL AI/ML Platform - Comprehensive",
        "version": "3.0.0",
        "total_players": len(mock_players),
        "total_predictions": len(mock_predictions),
        "uptime": "Online",
        "comprehensive_ai": {
            "status": "active",
            "knowledge_topics": len(comprehensive_football_knowledge),
            "coverage": "Complete NFL knowledge base"
        },
        "features": {
            "comprehensive_predictions": True,
            "intelligent_chat": True,
            "real_time_analytics": True,
            "comprehensive_football_knowledge": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
