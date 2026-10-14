"""
SignBridge AI - Predictive Smart Text & Word Autocomplete Engine
Provides prefix-based word completion, n-gram next-word prediction,
and abbreviation expansion for rapid gestural communication.
"""

from typing import List, Dict, Optional, Tuple


# Top frequent conversational English words (ordered by utility and frequency)
COMMON_WORDS: List[str] = [
    # Greetings & Courtesies
    "HELLO", "HI", "HEY", "GOODBYE", "BYE", "PLEASE", "THANK", "THANKS", "WELCOME",
    "SORRY", "EXCUSE", "PARDON", "CONGRATULATIONS",

    # Essential Actions & Needs (High Priority for Deaf/Mute Assistance)
    "HELP", "NEED", "WANT", "WATER", "FOOD", "MEDICINE", "DOCTOR", "HOSPITAL",
    "PAIN", "SICK", "HURT", "EMERGENCY", "POLICE", "AMBULANCE", "BATHROOM", "RESTROOM",
    "LIKE", "LOVE", "KNOW", "THINK", "UNDERSTAND",
    "SEE", "HEAR", "FEEL", "GO", "COME", "STAY", "WAIT", "STOP",
    "TALK", "SPEAK", "SIGN", "CALL", "WRITE", "READ", "LISTEN",
    "EAT", "DRINK", "SLEEP", "WAKE", "WALK", "DRIVE", "WORK", "PLAY",
    "LEARN", "TEACH", "MEET", "FIND", "GIVE", "TAKE", "MAKE", "LOOK",
    
    # Common Pronouns & Auxiliaries
    "I", "YOU", "HE", "SHE", "IT", "WE", "THEY", "ME", "MY", "YOUR", "HIS", "HER", "OUR", "THEIR",
    "AM", "IS", "ARE", "WAS", "WERE", "BE", "BEEN", "BEING",
    "HAVE", "HAS", "HAD", "DO", "DOES", "DID",
    "CAN", "COULD", "WILL", "WOULD", "SHALL", "SHOULD", "MAY", "MIGHT", "MUST",
    
    # Question Words
    "WHAT", "WHERE", "WHEN", "WHY", "WHO", "WHICH", "HOW",
    
    # People & Relations
    "FAMILY", "FRIEND", "MOTHER", "FATHER", "BROTHER", "SISTER", "SON", "DAUGHTER",
    "BABY", "CHILD", "PARENT", "TEACHER", "STUDENT", "PERSON", "PEOPLE", "NAME",
    "DEAF", "MUTE", "HEARING", "INTERPRETER",
    
    # Time & Days
    "TODAY", "TOMORROW", "YESTERDAY", "NOW", "LATER", "SOON", "ALWAYS", "NEVER",
    "MORNING", "AFTERNOON", "EVENING", "NIGHT", "TIME", "DAY", "WEEK", "MONTH", "YEAR",
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
    
    # Common Descriptors
    "YES", "NO", "NOT", "GOOD", "BAD", "GREAT", "BEAUTIFUL", "NICE", "KIND",
    "BIG", "SMALL", "FAST", "SLOW", "EASY", "HARD", "NEW", "OLD",
    "RIGHT", "WRONG", "TRUE", "SAME", "DIFFERENT", "READY", "BUSY",
    "MORE", "LESS", "MUCH", "MANY", "ALL", "SOME", "ANY", "NONE",
    
    # Locations & Objects
    "HOME", "HOUSE", "SCHOOL", "OFFICE", "STORE", "ROOM", "CAR", "BUS", "TRAIN",
    "PHONE", "COMPUTER", "BOOK", "PEN", "PAPER", "MONEY", "DOOR", "WINDOW",
    
    # Connectors & Prepositions
    "AND", "OR", "BUT", "BECAUSE", "SO", "IF", "WITH", "WITHOUT",
    "IN", "ON", "AT", "TO", "FROM", "FOR", "OF", "ABOUT", "OVER", "UNDER",
    "HERE", "THERE", "EVERYWHERE", "NOWHERE"
]

# Quick abbreviations and text acronyms commonly used in fast typing/chat
ABBREVIATIONS: Dict[str, str] = {
    "TY": "THANK YOU",
    "THX": "THANKS",
    "HRU": "HOW ARE YOU",
    "GM": "GOOD MORNING",
    "GN": "GOOD NIGHT",
    "GA": "GOOD AFTERNOON",
    "GE": "GOOD EVENING",
    "WBU": "WHAT ABOUT YOU",
    "IDK": "I DON'T KNOW",
    "SOS": "EMERGENCY NEED HELP",
    "PLS": "PLEASE",
    "PLZ": "PLEASE",
    "ILY": "I LOVE YOU",
    "NVM": "NEVER MIND",
    "BRB": "BE RIGHT BACK",
    "NP": "NO PROBLEM",
    "YW": "YOU'RE WELCOME",
    "TC": "TAKE CARE",
    "CU": "SEE YOU",
    "GL": "GOOD LUCK",
    "IMO": "IN MY OPINION",
    "OMW": "ON MY WAY",
    "ASAP": "AS SOON AS POSSIBLE",
    "RN": "RIGHT NOW",
    "BTW": "BY THE WAY",
    "OK": "OKAY",
    "K": "OKAY",
}

# Next-word associations (Bigram / Trigram predictions)
NEXT_WORD_PREDICTIONS: Dict[str, List[str]] = {
    "HELLO": ["HOW", "MY", "I", "FRIEND", "EVERYONE", "THERE"],
    "HI": ["HOW", "MY", "THERE", "FRIEND", "EVERYONE"],
    "HEY": ["HOW", "THERE", "WHAT", "FRIEND"],
    "HOW": ["ARE", "IS", "DO", "CAN", "WAS"],
    "HOW ARE": ["YOU", "THINGS", "WE", "THEY"],
    "THANK": ["YOU", "SO"],
    "THANK YOU": ["VERY", "SO", "FOR", "AGAIN"],
    "THANKS": ["A", "FOR", "SO"],
    "I": ["AM", "NEED", "WANT", "HAVE", "LIKE", "LOVE", "CAN", "DONT", "THINK", "KNOW", "FEEL"],
    "I AM": ["FINE", "GOOD", "HAPPY", "TIRED", "DEAF", "MUTE", "READY", "SORRY", "HERE"],
    "I NEED": ["HELP", "WATER", "FOOD", "MEDICINE", "DOCTOR", "TIME", "ASSISTANCE"],
    "I WANT": ["WATER", "FOOD", "TO", "HELP", "MORE"],
    "I LOVE": ["YOU", "MY", "THIS"],
    "I DONT": ["KNOW", "UNDERSTAND", "WANT", "HAVE", "LIKE"],
    "MY": ["NAME", "FAMILY", "FRIEND", "MOTHER", "FATHER", "PHONE", "HOME"],
    "MY NAME": ["IS"],
    "WHAT": ["IS", "ARE", "TIME", "DO", "HAPPENED"],
    "WHERE": ["IS", "ARE", "DO", "CAN"],
    "WHERE IS": ["THE", "MY", "BATHROOM", "RESTROOM", "DOCTOR", "HOSPITAL"],
    "WHEN": ["IS", "WILL", "CAN", "ARE"],
    "WHO": ["IS", "ARE", "CAN"],
    "WHY": ["ARE", "IS", "DID"],
    "PLEASE": ["HELP", "WAIT", "CALL", "COME", "GIVE", "TELL", "REPEAT"],
    "GOOD": ["MORNING", "AFTERNOON", "EVENING", "NIGHT", "JOB", "LUCK", "IDEA", "DAY"],
    "NICE": ["TO", "DAY", "MEETING"],
    "NICE TO": ["MEET"],
    "NICE TO MEET": ["YOU"],
    "SEE": ["YOU", "IT", "THAT"],
    "SEE YOU": ["LATER", "SOON", "TOMORROW"],
    "TAKE": ["CARE", "TIME", "THIS"],
    "YOU": ["ARE", "CAN", "WANT", "HAVE", "KNOW"],
    "YOU ARE": ["WELCOME", "KIND", "NICE", "GOOD"],
    "NO": ["PROBLEM", "WORRIES", "THANK"],
    "NO PROBLEM": ["AT"],
    "CAN": ["YOU", "I", "WE"],
    "CAN YOU": ["HELP", "HEAR", "SEE", "REPEAT", "WAIT"],
    "CALL": ["DOCTOR", "POLICE", "AMBULANCE", "MY", "FAMILY"],
    "ARE": ["YOU", "WE", "THEY"],
    "ARE YOU": ["OKAY", "FINE", "READY", "SURE", "BUSY"]
}


class PredictiveTextEngine:
    """
    Fast in-memory autocomplete and predictive text engine tailored for ASL sign-to-text.
    """

    def __init__(self, max_suggestions: int = 5):
        self.max_suggestions = max_suggestions
        self.common_words = COMMON_WORDS
        self.abbreviations = ABBREVIATIONS
        self.next_word_map = NEXT_WORD_PREDICTIONS

    def get_suggestions(self, current_text: str) -> List[str]:
        """
        Returns ranked word suggestions given the current accumulated text buffer.

        Cases handled:
        1. User has an active prefix (e.g. text="HELLO HO" -> prefix="HO"):
           Returns prefix completions (e.g. ["HOW", "HOME", "HOUSE"]).
        2. User typed an abbreviation (e.g. prefix="TY"):
           Returns expansion ["THANK YOU", ...].
        3. Text ends in space (e.g. text="HELLO "):
           Returns contextual next-word suggestions (e.g. ["HOW", "MY", "I", "FRIEND"]).
        4. Text is empty:
           Returns common opening words (e.g. ["HELLO", "HI", "I", "PLEASE"]).
        """
        stripped_raw = current_text.upper()
        if not stripped_raw:
            return ["HELLO", "HI", "I", "PLEASE", "HELP"][:self.max_suggestions]

        # Check if currently inside a word or just finished a word
        ends_with_space = current_text.endswith(" ")
        tokens = stripped_raw.split()

        if not ends_with_space and tokens:
            # Active prefix in progress
            prefix = tokens[-1]
            return self._get_prefix_completions(prefix)
        else:
            # Word boundary: suggest next words based on preceding context
            return self._get_next_word_suggestions(tokens)

    def _get_prefix_completions(self, prefix: str) -> List[str]:
        """Finds completions matching the active prefix."""
        suggestions: List[str] = []

        # 1. Check exact abbreviation match (e.g. TY -> THANK YOU)
        if prefix in self.abbreviations:
            suggestions.append(self.abbreviations[prefix])

        # 2. Check abbreviations starting with this prefix
        for abbr, expansion in self.abbreviations.items():
            if abbr.startswith(prefix) and expansion not in suggestions:
                suggestions.append(expansion)
                if len(suggestions) >= self.max_suggestions:
                    return suggestions

        # 3. Longer word completions from conversational vocabulary
        for word in self.common_words:
            if word.startswith(prefix) and word != prefix and word not in suggestions:
                suggestions.append(word)
                if len(suggestions) >= self.max_suggestions:
                    break

        # 4. Exact word match (e.g. if prefix is already a complete word like "HELLO" or "HE")
        if prefix in self.common_words and prefix not in suggestions:
            suggestions.append(prefix)

        return suggestions[:self.max_suggestions]

    def _get_next_word_suggestions(self, tokens: List[str]) -> List[str]:
        """Predicts high-probability next words based on recent token history."""
        if not tokens:
            return ["HELLO", "HI", "I", "PLEASE"][:self.max_suggestions]

        suggestions: List[str] = []

        # Check 3-word phrase context
        if len(tokens) >= 3:
            tri_key = f"{tokens[-3]} {tokens[-2]} {tokens[-1]}"
            if tri_key in self.next_word_map:
                for w in self.next_word_map[tri_key]:
                    if w not in suggestions:
                        suggestions.append(w)

        # Check 2-word phrase context
        if len(tokens) >= 2 and len(suggestions) < self.max_suggestions:
            bi_key = f"{tokens[-2]} {tokens[-1]}"
            if bi_key in self.next_word_map:
                for w in self.next_word_map[bi_key]:
                    if w not in suggestions:
                        suggestions.append(w)

        # Check 1-word context
        if len(tokens) >= 1 and len(suggestions) < self.max_suggestions:
            uni_key = tokens[-1]
            if uni_key in self.next_word_map:
                for w in self.next_word_map[uni_key]:
                    if w not in suggestions:
                        suggestions.append(w)

        # Fallback to general frequent starters if insufficient context
        defaults = ["YOU", "I", "HELP", "THANK", "PLEASE", "GOOD"]
        for d in defaults:
            if len(suggestions) >= self.max_suggestions:
                break
            if d not in suggestions and (not tokens or d != tokens[-1]):
                suggestions.append(d)

        return suggestions[:self.max_suggestions]

    def expand_abbreviation(self, token: str) -> Optional[str]:
        """Returns full expansion if token matches a registered shortcut."""
        return self.abbreviations.get(token.upper())
