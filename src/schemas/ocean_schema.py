
ocean_schema = {
    "title": "OceanProfile",
    "description": "OCEAN personality profile model",
    "type": "object",
    "properties": {
        "openness": {
            "title": "Openness",
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "Openness trait score (0-10)"
        },
        "conscientiousness": {
            "title": "Conscientiousness",
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "Conscientiousness trait score (0-10)"
        },
        "extraversion": {
            "title": "Extraversion",
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "Extraversion trait score (0-10)"
        },
        "agreeableness": {
            "title": "Agreeableness",
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "Agreeableness trait score (0-10)"
        },
        "neuroticism": {
            "title": "Neuroticism",
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "Neuroticism trait score (0-10)"
        }
    },
    "required": [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism"
    ]
}

