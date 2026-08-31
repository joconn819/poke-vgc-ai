"""Configuration and constants for Pokemon VGC formats."""

from dataclasses import dataclass, field
from typing import Set, Tuple
from enum import Enum


class Generation(Enum):
    """Pokemon generation enum."""
    GEN_8 = 8
    GEN_9 = 9
    GEN_10 = 10


@dataclass
class RegulationConfig:
    """Configuration for a VGC regulation.
    
    Defines legal Pokemon, items, abilities, moves, and tera types
    for a specific regulation.
    """
    regulation_id: str
    generation: Generation
    
    # Legality constraints
    legal_pokemon: Set[str] = field(default_factory=set)
    legal_items: Set[str] = field(default_factory=set)
    legal_moves: Set[str] = field(default_factory=set)
    legal_abilities: Set[str] = field(default_factory=set)
    legal_tera_types: Set[str] = field(default_factory=set)
    
    # Restrictions
    restricted_pokemon: Set[str] = field(default_factory=set)
    max_restricted_per_team: int = 1
    min_level: int = 1
    max_level: int = 100
    
    # Format specifics
    team_size: int = 6
    active_pokemon: int = 2
    is_vgc_doubles: bool = True
    
    def __hash__(self):
        """Make hashable for caching."""
        return hash(self.regulation_id)
    
    def is_pokemon_legal(self, pokemon_name: str) -> bool:
        """Check if a Pokemon is legal under this regulation."""
        return pokemon_name in self.legal_pokemon
    
    def is_item_legal(self, item_name: str) -> bool:
        """Check if an item is legal under this regulation."""
        return item_name in self.legal_items
    
    def is_move_legal(self, move_name: str) -> bool:
        """Check if a move is legal under this regulation."""
        return move_name in self.legal_moves


# Example regulation for testing (minimal)
REGULATION_SV2024_1 = RegulationConfig(
    regulation_id="sv2024-1",
    generation=Generation.GEN_9,
    legal_pokemon={
        "pikachu", "charizard", "dragonite", "salamence", "garchomp",
        "landorus", "tornadus", "thundurus", "torkoal", "urshifu",
        "calyrex", "regieleki", "regidrago", "glastrier", "spectrier",
        "kyurem", "zekrom", "reshiram", "xerneas", "yveltal",
        "groudon", "kyogre", "rayquaza", "blaziken", "venusaur",
        "rillaboom", "incineroar",
    },
    legal_items={
        "choice-band", "choice-scarf", "choice-specs", "assault-vest",
        "life-orb", "rocky-helmet", "air-balloon", "heavy-duty-boots",
    },
    legal_moves={
        "earthquake", "protected", "surf", "thunderbolt", "ice-beam",
        "shadow-ball", "focus-blast", "power-whip", "stone-edge",
        "outrage", "superpower", "heat-wave", "recover", "earth-power",
        "grassy-glide", "close-combat", "knock-off", "flare-blitz",
        "ice-punch",
    },
    legal_abilities={
        "pressure", "static", "rough-skin", "levitate", "speed-boost",
        "intimidate", "drought", "drizzle", "grassy-surge", "chilling-neigh",
        "unseen-hand",
    },
    legal_tera_types={
        "normal", "fire", "water", "electric", "grass", "ice", "fighting",
        "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
        "dragon", "dark", "steel", "fairy",
    },
    restricted_pokemon={"groudon", "kyogre", "rayquaza"},
)


# Champions MB Regulation (Sword/Shield era, Level 50 Doubles)
# Based on VGC 2020 Champions League regulation data
# Includes Galar regional dex (Pokemon 1-400) plus Isle of Armor and Crown Tundra DLC
REGULATION_CHAMPIONS_MB = RegulationConfig(
    regulation_id="champions-mb",
    generation=Generation.GEN_8,
    min_level=50,
    max_level=50,
    legal_pokemon={
       # Galar Starter lines
       "grookey", "thwackey", "rillaboom",
       "scorbunny", "raboot", "cinderace",
       "sobble", "drizzile", "inteleon",
        
       # Early game regulars
       "pidgeot", "pidgeotto", "pidgeot-gmax",
       "pikachu", "raichu", "raichu-galar",
        
       # Corviknight line
       "rookidee", "corvisquire", "corviknight", "corviknight-gmax",
        
       # Duraludon line
       "duraludon", "duraludon-gmax",
        
       # Dragapult line
       "dreepy", "drakloak", "dragapult", "dragapult-gmax",
        
       # Fossil Pokemon (resurrected in Galar)
       "dracozolt", "arctozolt", "dracovish", "arctovish",
        
       # Applin evolution line
       "applin", "flapple", "appletun",
        
       # Sword/Shield exclusives & regulars
       "wooloo", "dubwool",
       "electric-type-common",
       "magnemite", "magneton", "magnezone",
       "rotom", "rotom-heat", "rotom-wash", "rotom-frost", "rotom-fan", "rotom-mow",
       "ditto",
       "snorunt", "glalie", "froslass",
       "delibird",
       "swinub", "piloswine", "mamoswine",
       "sneasel", "weavile",
       "timburr", "gurdurr", "conkeldurr",
       "woobat", "swoobat",
       "dwebble", "crustle",
       "sigilyph",
       "emolga",
       "joltik", "galvantula",
       "ferroseed", "ferrothorn",
       "klink", "klang", "klinklang",
       "vanillite", "vanilish", "vanilluxe",
       "duskull", "dusclops", "dusknoir",
       "porygon", "porygon2", "porygon-z",
       "beldum", "metang", "metagross",
       "dreepy", "drakloak", "dragapult",
       "shelmet", "accelgor",
       "venomoth", "venomoth-galar",
       "arcanine", "arcanine-galar",
       "growlithe", "growlithe-galar",
       "vulpix", "vulpix-alola", "ninetales", "ninetales-alola",
       "mankey", "mankey-alola", "primeape", "primeape-alola",
       "diglett", "diglett-alola", "dugtrio", "dugtrio-alola",
       "farfetchd", "farfetchd-galar",
       "psyduck", "golduck",
       "slowpoke", "slowpoke-galar", "slowbro", "slowbro-galar", "slowking", "slowking-galar",
       "seel", "dewgong",
       "shellder", "cloyster",
       "krabby", "kingler", "kingler-gmax",
       "exeggcute", "exeggcute-alola", "exeggutor", "exeggutor-alola",
       "cubone", "cubone-alola", "marowak", "marowak-alola",
       "rhyhorn", "rhydon", "rhyperior",
       "happiny", "chansey", "blissey",
       "munna", "musharna",
       "wooper",
       "phanpy", "donphan",
       "mantyke", "mantine",
       "wailmer", "wailord",
       "barboach", "whiscash",
       "corphish", "crawdaunt",
       "feebas", "milotic",
       "carvanha", "sharpedo",
       "wailmer", "wailord",
       "barboach", "whiscash",
       "corphish", "crawdaunt",
       "feebas", "milotic",
       "carvanha", "sharpedo",
       "trapinch", "vibrava", "flygon",
       "lileep", "cradily",
       "anorith", "armaldo",
       "cranidos", "rampardos",
       "shieldon", "bastionage",
        
       # Pokemon returning via DLC
       "horsea", "seadra", "kingdra",
       "bagon", "shelgon", "salamence", "salamence-gmax",
       "beldum", "metang", "metagross", "metagross-gmax",
       "jangmo-o", "hakamo-o", "kommo-o",
       "goomy", "sliggoo", "goodra",
       "deino", "zweilous", "hydreigon",
        
       # Legendary & Mythical (many available in Crown Tundra)
       "articuno", "articuno-galar",
       "zapdos", "zapdos-galar",
       "moltres", "moltres-galar",
       "mewtwo",
       "raikou", "entei", "suicune",
       "lugia", "ho-oh",
       "kyogre", "groudon", "rayquaza",
       "regice", "regirock", "registeel", "regigigas",
       "latias", "latios",
       "tornadus", "tornadus-therian",
       "thundurus", "thundurus-therian",
       "landorus", "landorus-therian",
       "enamorus", "enamorus-therian",
       "reshiram", "zekrom", "kyurem",
       "xerneas", "yveltal", "zygarde",
       "diancie", "hoopa", "volcanion",
        
       # Crown Tundra DLC legendaries
       "calyrex", "calyrex-shadow", "calyrex-ice",
       "glastrier", "spectrier",
       "urshifu", "urshifu-rapid-strike", "urshifu-gmax", "urshifu-rapid-strike-gmax",
       "kubfu",
        
       # Isle of Armor & Crown Tundra additions (Alola forms, Galar forms, etc.)
       "diglett-alola", "dugtrio-alola",
       "rattata", "raticate", "raticate-galar",
       "raichu-alola",
       "vulpix-alola", "ninetales-alola",
       "growlithe-galar", "arcanine-galar",
       "wigglytuff", "jigglypuff",
       "oddish", "gloom", "vileplume", "bellossom",
       "paras", "parasect",
       "venonat", "venomoth", "venomoth-galar",
       "bellsprout", "weepinbell", "victreebel",
       "tentacool", "tentacruel",
       "slowpoke-galar", "slowbro-galar", "slowking-galar",
       "seel", "dewgong",
       "shellder", "cloyster",
       "gastly", "haunter", "gengar", "gengar-gmax",
       "onix", "steelix", "steelix-gmax",
       "drowzee", "hypno",
       "krabby", "kingler", "kingler-gmax",
       "voltorb", "electrode", "electrode-galar",
       "exeggcute-alola", "exeggutor-alola",
       "marowak-alola",
       "weezing", "weezing-galar",
       "rhyhorn", "rhydon", "rhyperior",
       "chansey", "blissey",
       "tangela", "tangrowth",
       "kangaskhan", "kangaskhan-gmax",
       "horsea", "seadra", "kingdra",
       "goldeen", "seaking",
       "staryu", "starmie",
       "mr-mime", "mr-mime-galar", "mr-rime",
       "cloyster",
       "lapras", "lapras-gmax",
       "snorlax", "snorlax-gmax",
       "articuno", "zapdos", "moltres",
       "ditto",
       "omanyte", "omastar",
       "kabuto", "kabutops",
       "aerodactyl", "aerodactyl-gmax",
       "sycther", "scizor", "scizor-gmax",
       "jynx", "smoochum",
       "electabuzz", "electivire",
       "magby", "magnemite", "magneton", "magnezone",
       "pinsir", "pinsir-gmax",
       "tauros",
       "lapras",
       "ditto",
       "eevee", "eevee-gmax",
       "vaporeon", "jolteon", "flareon", "espeon", "umbreon", "leafeon", "glaceon", "sylveon",
       "porygon", "porygon2", "porygon-z",
       "seel", "dewgong",
       "shellder", "cloyster",
       "krabby", "kingler",
        
       # Hoenn Pokemon (returning in Crown Tundra)
       "treecko", "grovyle", "sceptile", "sceptile-gmax",
       "torchic", "combusken", "blaziken", "blaziken-gmax",
       "mudkip", "marshtomp", "swampert", "swampert-gmax",
       "poochyena", "mightyena",
       "zigzagoon", "zigzagoon-galar", "linoone", "linoone-galar", "obstagoon",
       "taillow", "swellow",
       "wingull", "pelipper",
       "ralts", "kirlia", "gardevoir", "gardevoir-gmax", "gallade",
       "surskit", "masquerain",
       "shroomish", "breloom", "breloom-gmax",
       "slakoth", "vigoroth", "slaking",
       "whismur", "loudred", "exploud",
       "skitty", "delcatty",
       "spinda",
       "cacnea", "cacturne",
       "swablu", "altaria", "altaria-gmax",
       "zangoose",
       "castform",
       "kecleon",
       "shuppet", "banette", "banette-gmax",
       "duskull", "dusclops", "dusknoir",
       "tropius",
       "chimecho", "chimera",
       "absol", "absol-gmax",
       "wynaut", "wobbuffet",
       "snorunt", "glalie", "froslass",
       "spheal", "sealeo", "walrein",
       "carvanha", "sharpedo", "sharpedo-gmax",
       "wailmer", "wailord",
       "barboach", "whiscash",
       "corphish", "crawdaunt",
       "feebas", "milotic",
       "carvanha", "sharpedo",
       "trapinch", "vibrava", "flygon",
       "lileep", "cradily",
       "anorith", "armaldo",
       "feebas", "milotic",
       "sharpedo", "sharpedo-mega",
       "wailord",
       "numel", "camerupt", "camerupt-gmax",
       "torkoal", "torkoal-gmax",
       "spoink", "grumpig",
       "spinda",
       "trapinch", "vibrava", "flygon",
       "cacnea", "cacturne",
       "swablu", "altaria", "altaria-gmax",
       "zangoose",
       "castform",
       "kecleon",
       "shuppet", "banette", "banette-gmax",
       "duskull", "dusclops", "dusknoir",
       "tropius",
       "chimecho", "chimera",
       "absol", "absol-gmax",
       "wynaut", "wobbuffet",
        
       # Sinnoh Pokemon (returning via Crown Tundra)
       "turtwig", "grotle", "torterra", "torterra-gmax",
       "chimchar", "monferno", "infernape", "infernape-gmax",
       "piplup", "prinplup", "empoleon", "empoleon-gmax",
       "starly", "staravia", "staraptor",
       "bidoof", "bibarel",
       "buneary", "lopunny", "lopunny-gmax",
       "glameow", "purugly",
       "chatot",
       "spiritomb",
       "gible", "gabite", "garchomp", "garchomp-gmax",
       "munchlax", "snorlax", "snorlax-gmax",
       "riolu", "lucario", "lucario-gmax",
       "happiny", "chansey", "blissey",
       "mantyke", "mantine",
       "cranidos", "rampardos",
       "shieldon", "bastionage",
       "snover", "abomasnow", "abomasnow-gmax",
       "rotom", "rotom-heat", "rotom-wash", "rotom-frost", "rotom-fan", "rotom-mow",
       "uxie", "mesprit", "azelf",
       "dialga", "palkia", "giratina", "giratina-origin",
       "cresselia",
        
       # Unova Pokemon (returning via Crown Tundra)
       "snivy", "servine", "serperior", "serperior-gmax",
       "tepig", "pignite", "emboar", "emboar-gmax",
       "oshawott", "dewott", "samurott", "samurott-gmax",
       "patrat", "watchog",
       "pidove", "tranquill", "unfezant",
       "blitzle", "zebstrika",
       "roggenrola", "boldore", "gigalith", "gigalith-gmax",
       "woobat", "swoobat",
       "dwebble", "crustle",
       "sigilyph",
       "emolga",
       "joltik", "galvantula",
       "ferroseed", "ferrothorn", "ferrothorn-gmax",
       "klink", "klang", "klinklang",
       "axew", "fraxure", "haxorus", "haxorus-gmax",
       "cubchoo", "beartic", "beartic-gmax",
       "shelmet", "accelgor",
       "stunfisk", "stunfisk-galar",
       "mienfoo", "mienshao",
       "rufflet", "braviary", "braviary-galar",
       "vullaby", "mandibuzz",
       "druddigon",
       "golett", "golurk", "golurk-gmax",
       "pawniard", "pawmotzal", "kingambit",
       "archen", "archeops",
       "zorua", "zoroark",
       "minccino", "cinccino",
       "deerling", "sawsbuck",
       "foongus", "amoonguss",
       "ferroseed", "ferrothorn",
       "klink", "klang", "klinklang",
       "joltik", "galvantula",
       "tynamo", "eelektrik", "eelektross",
       "elgyem", "beheeyem",
       "litwick", "lampent", "chandelure", "chandelure-gmax",
       "axew", "fraxure", "haxorus", "haxorus-gmax",
       "cubchoo", "beartic", "beartic-gmax",
       "shelmet", "accelgor",
       "stunfisk", "stunfisk-galar",
       "mienfoo", "mienshao",
       "rufflet", "braviary", "braviary-galar",
       "vullaby", "mandibuzz",
       "druddigon",
       "golett", "golurk", "golurk-gmax",
       "pawniard", "pawmotzal", "kingambit",
        
       # Kalos Pokemon (returning via Crown Tundra)
       "chespin", "quilladin", "chesnaught", "chesnaught-gmax",
       "fennekin", "braixen", "delphox", "delphox-gmax",
       "froakie", "frogadier", "greninja", "greninja-ash", "greninja-gmax",
       "fletchling", "fletchinder", "talonflame", "talonflame-gmax",
       "bunnelby", "diggersby", "diggersby-gmax",
       "scatterbug", "spewpa", "vivillon",
       "litleo", "pyroar",
       "flabébé", "floette", "florges", "florges-gmax",
       "skiddo", "gogoat",
       "pancham", "pangoro",
       "furfrou",
       "espurr", "meowstic",
       "honedge", "doublade", "aegislash", "aegislash-gmax",
       "spritzee", "aromatisse",
       "swirlix", "slurpuff", "slurpuff-gmax",
       "inkay", "malamar",
       "binacle", "barbaracle",
       "helioptile", "heliolisk",
       "tyrunt", "tyrantrum", "tyrantrum-gmax",
       "amaura", "aurorus",
       "sylveon",
       "hawlucha",
       "goomy", "sliggoo", "goodra",
       "jangmo-o", "hakamo-o", "kommo-o", "kommo-o-gmax",
       "phantump", "trevenant", "trevenant-gmax",
       "pumpkaboo", "gourgeist",
       "bergmite", "avalugg",
       "noibat", "noivern", "noivern-gmax",
       "xerneas",
       "yveltal",
       "zygarde",
       "diancie",
       "hoopa", "hoopa-unbound",
       "volcanion",
        
       # Alola Pokemon (many returning via DLC)
       "rowlet", "dartrix", "decidueye", "decidueye-galar",
       "litten", "torracat", "incineroar", "incineroar-gmax",
       "popplio", "brionne", "primarina", "primarina-gmax",
       "pikipek", "trumbeak", "toucannon",
       "yungoos", "gumshoos", "gumshoos-totem",
       "crabrawler", "crabominable",
       "oricorio",
       "cutiefly", "ribombee",
       "rockruff", "rockruff-dusk", "lycanroc", "lycanroc-midnight", "lycanroc-dusk",
       "wishiwashi",
       "mareanie", "toxapex",
       "mudbray", "mudsdale",
       "dewpider", "araquanid",
       "fomantis", "lurantis",
       "morelull", "shiinotic",
       "salandit", "salazzle", "salazzle-gmax",
       "stufful", "bewear", "bewear-gmax",
       "bounsweet", "steenee", "tsareena", "tsareena-gmax",
       "wimpod", "golisopod", "golisopod-gmax",
       "sandygast", "palossand",
       "pyukumuku",
       "type-null", "silvally", "silvally-gmax",
       "komala",
       "turtonator",
       "togedemaru",
       "mimikyu", "mimikyu-gmax",
       "bruxish",
       "drampa", "drampa-gmax",
       "dhelmise",
       "jangmo-o", "hakamo-o", "kommo-o", "kommo-o-gmax",
       "cosmog", "cosmoem", "solgaleo", "lunala",
       "nihilego",
       "buzzwole",
       "pheromosa",
       "xurkitree",
       "celesteela",
       "kartana",
       "guzzlord",
       "poipole", "naganadel",
       "stakataka",
       "blacephalon",
       "magearna",
       "marshadow",
       "zeraora",
    },
    legal_items={
       # Generic healing items
       "adrenaline-orb",
        
       # Stat-boosting held items
       "choice-band", "choice-scarf", "choice-specs",
       "assault-vest",
       "weakness-policy",
        
       # Damage/Offensive items
       "life-orb",
       "expert-belt",
       "entei-orb",  # If applicable
        
       # Defensive items
       "rocky-helmet",
       "air-balloon",
       "heavy-duty-boots",
       "utility-umbrella",
        
       # Weather-related items
       "heat-rock",
       "damp-rock",
       "smooth-rock",
       "icy-rock",
        
       # Ability-related
       "eviolite",
       "eject-pack",
        
       # Type-boost items
       "charcoal", "mystic-water", "miracle-seed", "magnet", "twister", "silk-scarf",
       "metallic-plate",  # Gen 8
        
       # Stat items
       "muscle-band", "wise-glasses",
        
       # Luck items (not competitive but legal)
       "luck-incense",
        
       # Berries (held)
       "occa-berry", "passho-berry", "wacan-berry", "rindo-berry", "yache-berry",
       "cheri-berry", "chesto-berry", "pecha-berry", "rawst-berry", "aspear-berry",
       "lum-berry", "persim-berry", "sitrus-berry",
        
       # Evolution stones/misc
       "assault-vest",
    },
    legal_moves={
       # Physical moves
       "earthquake", "stone-edge", "power-whip", "close-combat", "superpower",
       "knock-off", "flare-blitz", "aqua-jet", "icicle-crash", "iron-head",
       "play-rough", "moonblast", "outrage", "draco-meteor", "zen-headbutt",
       "shadow-claw", "shadow-punch", "brick-break", "focus-blast", "poison-powder",
       "thunder-punch", "ice-punch", "fire-punch", "aqua-punch",
       "u-turn", "dual-wingbeat", "grassy-glide", "surging-strikes", "close-combat",
        
       # Special moves
       "surf", "thunderbolt", "ice-beam", "shadow-ball", "focus-blast",
       "heat-wave", "earth-power", "psychic", "power-gem", "flash-cannon",
       "dark-pulse", "sludge-bomb", "poison-powder", "powder-snow", "water-spout",
       "discharge", "volt-switch", "thundershock", "thunder-wave",
        
       # Support/utility
       "protect", "recover", "swords-dance", "calm-mind", "dragon-dance",
       "nasty-plot", "bulk-up", "work-up", "tailwind", "trick-room",
       "light-screen", "reflect", "stealth-rock", "spikes", "toxic-spikes",
       "will-o-wisp", "leech-seed", "toxic", "swagger", "charm",
       "fake-out", "follow-me", "ally-switch", "rage-powder", "spotlight",
       "aurora-veil", "hail", "rain-dance", "sandstorm", "sunny-day",
       "grassy-terrain", "electric-terrain", "misty-terrain", "psychic-terrain",
       "terrain-pulse",
        
       # Priority moves
       "quick-attack", "aqua-jet", "ice-shard", "mach-punch", "shadow-sneak",
       "sucker-punch", "vital-throw",
        
       # Gen 8 specific
       "grassy-glide", "close-combat", "surging-strikes", "astral-barrage",
       "dual-wingbeat", "meteor-assault", "steel-beam",
    },
    legal_abilities={
       # Common stat-boosting
       "intimidate", "pressure", "static", "rough-skin", "speed-boost",
       "drizzle", "drought", "sand-stream", "snow-warning",
        
       # Weather-related
       "dry-skin", "water-absorb", "flash-fire", "volt-absorb", "lightning-rod",
       "storm-drain", "thick-fat",
        
       # Terrain-related
       "grassy-surge", "electric-surge", "misty-surge", "psychic-surge",
        
       # Damage-related
       "filter", "levitate", "marvel-scale", "regenerator",
        
       # Gen 8 new/relevant abilities
       "chilling-neigh", "grim-neigh",
        
       # Competitive staples
       "protean", "stance-change", "shadow-tag", "trick-or-treat",
       "unaware", "competitive", "defiant", "weak-armor", "download",
       "huge-power", "pure-power", "sheer-force", "iron-fist",
        
       # Utility
       "harvest", "effect-spore", "immunity", "comatose", "neutralizing-gas",
    },
    legal_tera_types=set(),  # No Tera in Gen 8
    restricted_pokemon={
       # Box legends
       "zacian", "zamazenta",
       # Crown Tundra legendaries (restricted)
       "kyogre", "groudon", "kyurem",
       "xerneas", "yveltal",
       "reshiram", "zekrom",
       "dialga", "palkia", "giratina",
       # Mewtwo
       "mewtwo",
    },
)
