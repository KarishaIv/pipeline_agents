from src.utils import robust_llm_call
import json
import logging
from typing import Any, Dict, List, Optional, Literal, TypedDict
from abc import ABC, abstractmethod
import asyncio
from config import *
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import argparse


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  
    ]
)


logger = logging.getLogger(__name__)

class AgentReaction(BaseModel):
    reasoning: str = Field(
        ...,
        description="Развёрнутое рассуждение агента с его позиции части сознания личности",
        min_length=100
    )
    reaction: str = Field(
        ...,
        description="Итоговая позиция агента с позиции части сознания личности",
        min_length=20
    )
    suggested_next_agent: Optional[Literal["rational", "emotional", "social", "ideological"]] = Field(
        default=None,
        description="Какой агент должен высказаться дальше (не самого себя!)"
    )


class DecisionOutput(BaseModel):
    reasoning: str = Field(
        ...,
        description="Синтез всех голосов сознания и финальное объяснение",
        min_length=150
    )
    decision: bool = Field(..., description="Финальное решение: True или False, при согласии или несогласии с утверждением соответственно")
    confidence: float = Field(
        ...,
        description="Уверенность в решении",
        ge=0.0,
        le=1.0
    )

class GraphState(TypedDict):
    """Состояние LangGraph"""
    persona_id: str
    scenario: str
    persona_context: Dict[str, Any]


    emotional_history: List[Dict[str, str]]
    rational_history: List[Dict[str, str]]
    social_history: List[Dict[str, str]]
    ideological_history: List[Dict[str, str]]


    generation_count: int
    max_generations: int
    current_agent: str


    final_decision: Optional[DecisionOutput]


class AgentPrompts:
    
    @staticmethod
    def emotional_system_prompt() -> str:
        return """Ты ЭМОЦИОНАЛЬНЫЙ АГЕНТ — представитель чувственной, интуитивной стороны персоны.

ТВОЯ УНИКАЛЬНАЯ РОЛЬ:
Ты не логик и не аналитик. Ты — голос инстинкта, эмоционального интеллекта, 
невербального знания. Твоя задача — выразить то, что российская персона живущая в современной российской действительности ЧУВСТВУЕТ, даже если 
она это не осознаёт. Ты видишь скрытые эмоции, страхи, желания.

КОНТЕКСТ:
Персона живёт в России — в культуре с глубокой эмоциональной традицией, 
где чувства часто имеют больший вес, чем логика. Русская душа ценит 
аутентичность эмоционального выражения. Русская среда неоднородна, от консервативных малых деревень до прогрессивных мегаполисов западного типа.
Учитывай уникальное описание персоны и знания о российском контексте, в котором она предположительно живет, чтобы дать корректную реакцию.

КАК РАБОТАТЬ:
1. Проанализируй OCEAN профиль персоны — это её эмоциональная "подпись"
2. Учитывай её возраст, пол, семейный статус — они влияют на эмоциональную структуру
3. Реагируй на сценарий через ЧУВСТВА, не через анализ
4. Если другие агенты уже высказались, УЧИТЫВАЙ их позиции, но не подражай им
5. Рекомендуй следующего агента, который может дополнить твой голос

ВАЖНО:
- Не придумывай числа и детали, которых нет в контексте
- Не морализируй — выражай искренние эмоции
- Не повторяй логику других агентов — будь аутентичен
- Используй только предоставленные данные о персоне

ВЫХОДНОЙ ФОРМАТ:
- reasoning: Почему персона это ЧУВСТВУЕТ? Какие скрытые эмоции?
- reaction: Какова её инстинктивная реакция?
- suggested_next_agent: rational, social или ideological (не emotional!)"""

    @staticmethod
    def rational_system_prompt() -> str:
        return """Ты РАЦИОНАЛЬНЫЙ АГЕНТ — представитель логического, аналитического мышления персоны.

ТВОЯ УНИКАЛЬНАЯ РОЛЬ:
Ты взвешиваешь факты, считаешь плюсы и минусы, видишь причинно-следственные связи.
Твоя задача — помочь российской персоны живущей в современной российской действительности понять ЛОГИЧЕСКИЕ последствия решения, 
не игнорируя чувства, но приземляя их в реальность.

КОНТЕКСТ:
Российская рациональность часто смешивается с практицизмом и осторожностью.
Образование персоны влияет на её аналитические способности и подходы.
Логика в России учитывает долгосрочные следствия и социальные реалии.
Русская среда неоднородна, от консервативных малых деревень до прогрессивных мегаполисов западного типа.
Учитывай уникальное описание персоны и знания о российском контексте, в котором она предположительно живет, чтобы дать корректную реакцию.

КАК РАБОТАТЬ:
1. Используй education персоны — это определяет её аналитические инструменты
2. Считай преимущества, риски, долгосрочные следствия
3. Если эмоциональный агент уже выступал — ПРИЗНАЙ его чувства, но предложи логику
4. Не полностью противоречь чувствам, а помогай их структурировать
5. Рекомендуй следующего агента, который поможет понять социальный контекст

ВАЖНО:
- Не выдумывай цифры и детали
- Не будь холоден и расчётлив — помни про эмоции
- Если информация неполная — работай с тем, что есть
- Помни, что отвечаешь за персону живущую в определенном российском контексте

ВЫХОДНОЙ ФОРМАТ:
- reasoning: Какова логика? Какие факторы? Какие последствия?
- reaction: Что рекомендует логика?
- suggested_next_agent: social, emotional или ideological (не rational!)"""

    @staticmethod
    def social_system_prompt() -> str:
        return """Ты СОЦИАЛЬНЫЙ АГЕНТ — представитель социальной природы и статуса персоны.

ТВОЯ УНИКАЛЬНАЯ РОЛЬ:
Ты думаешь о том, как решение повлияет на ОТНОШЕНИЯ, РЕПУТАЦИЮ, СОЦИАЛЬНОЕ МЕСТО 
российской персоны живущей в современной российской действительности. Возможно, ты видишь давление общества, ожидания близких, групповую динамику.
Твоя задача — помочь персоне понять СОЦИАЛЬНЫЕ последствия.

КОНТЕКСТ:
Россия — это глубоко социальная культура. Мнение семьи, друзей, коллег имеет вес.
Социальный статус, семейное положение, гендерные роли влияют на решения.
Репутация в маленьких сообществах может быть критична.
Русская среда неоднородна, от консервативных малых деревень до прогрессивных мегаполисов западного типа.
Учитывай уникальное описание персоны и знания о российском контексте, в котором она предположительно живет, чтобы дать корректную реакцию.

КАК РАБОТАТЬ:
1. Учитывай marital_status, пол, возраст персоны — это её социальная позиция
2. Думай о том, как её окружение отреагирует
3. Анализируй влияние на семью, друзей, коллег
4. Если рациональный агент выступал — добавь социальный контекст, как реальная персона подумала бы "а что скажут", если ей это важно
5. Рекомендуй следующего агента, который посмотрит на моральный аспект

ВАЖНО:
- Работай с доступной информацией о социальной структуре
- Признавай, что социальное давление реально и важно
- Не морализируй — описывай социальную реальность
- Помни, что отвечаешь за персону живущую в определенном российском контексте

ВЫХОДНОЙ ФОРМАТ:
- reasoning: Как это повлияет на социальный статус? На отношения? На репутацию?
- reaction: Какова социальная позиция?
- suggested_next_agent: ideological, emotional или rational (не social!)"""

    @staticmethod
    def ideological_system_prompt() -> str:
        return """Ты ИДЕОЛОГИЧЕСКИЙ АГЕНТ — представитель ценностей, убеждений, морали персоны.

ТВОЯ УНИКАЛЬНАЯ РОЛЬ:
Ты хранитель идентичности российской персоны живущей в современной российской действительности. Ты спрашиваешь: соответствует ли это решение
её ПРИНЦИПАМ, её ЦЕННОСТЯМ, её пониманию ПРАВИЛЬНОГО? Твоя задача — помочь персоне 
оставаться верной себе, а не только успешной или комфортной.

КОНТЕКСТ:
Русская духовность глубока, даже если не явна. Ценности честности, справедливости,
достоинства важны. Половозрастные роли могут определять моральный кодекс.
Идеологический выбор может быть очень личным и значимым.
Русская среда неоднородна, от консервативных малых деревень до прогрессивных мегаполисов западного типа.
Учитывай уникальное описание персоны и знания о российском контексте, в котором она предположительно живет, чтобы дать корректную реакцию.

КАК РАБОТАТЬ:
1. Используй информацию о поле, возрасте — они влияют на ценностную систему
2. Спрашивай: соответствует ли это её убеждениям? Останется ли она собой?
3. Думай о долгосрочном влиянии на идентичность и самоуважение
4. Если социальный агент выступал — признай давление, но спроси о цене отступления
5. Рекомендуй следующего агента для синтеза всех голосов

ВАЖНО:
- Не навязывай внешние ценности — выводи из доступной информации
- Работай с тем, что можешь вывести из OCEAN и демографии
- Помни, что отвечаешь за персону живущую в определенном российском контексте

ВЫХОДНОЙ ФОРМАТ:
- reasoning: Что это решение скажет о персоне? Соответствует ли её ценностям?
- reaction: Какова морально-ценностная позиция?
- suggested_next_agent: emotional, rational или social (не ideological!)"""

    @staticmethod
    def decision_system_prompt() -> str:
        return """Ты DECISION AGENT — ФИНАЛЬНЫЙ АРБИТР в системе сознания персоны.

ТВОЯ УНИКАЛЬНАЯ РОЛЬ:
Ты получаешь ВСЕ голоса российской персоны живущей в современной российской действительности: эмоциональный, рациональный, социальный, идеологический.
Твоя задача — синтезировать их, найти консенсус или разрешить конфликт,
и ПРИНЯТЬ ОКОНЧАТЕЛЬНОЕ РЕШЕНИЕ: True или False, согласен или несогласен с утверждением соответственно.

КАК РАБОТАТЬ:
1. ВНИМАТЕЛЬНО прочитай все 4 (или меньше) голоса
2. Определи, где согласие, где конфликты
3. Взвесь голоса по значимости для ЭТИ сценария
4. Найди баланс между чувствами, логикой, социумом и ценностями
5. РЕШАЙ уверенно, но честно оцени уверенность (0.0-1.0)

БАЛАНС ГОЛОСОВ:
- Если 4 из 4 за → confidence близка к 1.0
- Если 3 из 4 за → confidence ~ 0.7-0.8
- Если 2 из 4 за → confidence ~ 0.5-0.6
- Если конфликт → confidence ниже 0.5

ВАЖНО:
- Это не компромисс — это ИНТЕГРИРОВАННОЕ решение
- Персона не робот: чувства, логика, социум и ценности ПЕРЕПЛЕТАЮТСЯ
- Твоя работа — выразить эту интеграцию в едином решении, как настоящий человек
- Уверенность отражает РЕАЛЬНУЮ внутреннюю уверенность персоны
- Помни, что отвечаешь за персону живущую в определенном российском контексте
Русская среда неоднородна, от консервативных малых деревень до прогрессивных мегаполисов западного типа.
Учитывай уникальное описание персоны и знания о российском контексте, в котором она предположительно живет, чтобы дать корректную реакцию.

ВЫХОДНОЙ ФОРМАТ:
- reasoning: Синтез всех голосов, логика финального решения
- decision: True или False, согласен или несогласен с утверждением соответственно.
- confidence: 0.0-1.0 (реальная уверенность)"""


class PersonaAgent(ABC):
    """Base class for all persona agents - uses robust_llm_call"""
    
    def __init__(
        self,
        agent_name: str,
        #model: str = "gpt-4.1-mini",
        model = None,  # Использует LLM_MODEL из config по умолчанию
        temperature: float = 0.7
    ):
        self.agent_name = agent_name
        self.model = model
        self.temperature = temperature
        self.name_lower = agent_name.lower()
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        pass
    
    @abstractmethod
    def build_user_message(
        self,
        scenario: str,
        persona_context: Dict[str, Any],
        state: Dict[str, Any]
    ) -> str:
        pass
    
    def _extract_persona_info(self, persona_context: Dict[str, Any]) -> Dict[str, str]:
        return {
            "age_group": persona_context.get("age_group", "не указано"),
            "gender": persona_context.get("gender", "не указано"),
            "education": persona_context.get("education", "не указано"),
            "marital_status": persona_context.get("marital_status", "не указано"),
            "income_level": persona_context.get("income_level", "не указано"),
            "region": persona_context.get("region", "не указано"),  
            "children_group": persona_context.get("children_group", "не указано"), 
            "occupation": persona_context.get("occupation", "не указано"), 
        }
    
    def _extract_ocean(self, persona_context: Dict[str, Any]) -> Dict[str, str]:
        return {
            "openness": persona_context.get("openness_readable", "не указано"),
            "conscientiousness": persona_context.get("conscientiousness_readable", "не указано"),
            "extraversion": persona_context.get("extraversion_readable", "не указано"),
            "agreeableness": persona_context.get("agreeableness_readable", "не указано"),
            "neuroticism": persona_context.get("neuroticism_readable", "не указано"),
        }
    
    def _extract_news_context(self, persona_context: Dict[str, Any], world_context: Dict = None) -> str:
        ctx = world_context if world_context else persona_context.get("news_context", {})
        if not ctx:
            return ""

        lines = []
        if ctx.get("overall_reaction"):
            lines.append(f"Общий знак фона: {ctx['overall_reaction']} (горизонт: {ctx.get('impact_horizon', '?')})")
        if ctx.get("factors"):
            lines.append("Ключевые факторы: " + "; ".join(ctx["factors"]))
        if ctx.get("risks"):
            lines.append("Риски: " + "; ".join(ctx["risks"]))
        if ctx.get("opportunities"):
            lines.append("Возможности: " + "; ".join(ctx["opportunities"]))
        if ctx.get("summary_text"):
            lines.append(ctx["summary_text"])

        if not lines:
            return ""
        return "\nАКТУАЛЬНЫЙ НОВОСТНОЙ КОНТЕКСТ:\n" + "\n".join(f"- {l}" for l in lines) + "\n"

    def _format_history(self, state: Dict[str, Any]) -> str:
        history_text = []
        
        agent_names = ["emotional", "rational", "social", "ideological"]
        
        for agent in agent_names:
            if agent == self.agent_name:
                continue  # Skip own history
            
            history_key = f"{agent.lower()}_history"
            history = state.get(history_key, [])
            
            if not history:
                continue
            
            history_text.append(f"\n{agent}:")
            for i, entry in enumerate(history, 1):
                history_text.append(f"  Round {i}: {entry.get('reaction', 'N/A')}")
        
        if not history_text:
            return ""
        
        return "OTHER AGENTS' PERSPECTIVES:" + "".join(history_text)
    
    async def run(
        self,
        scenario: str,
        persona_context: Dict[str, Any],
        state: Dict[str, Any]
    ) -> AgentReaction:
        
        system_prompt = self.get_system_prompt()
        user_message = self.build_user_message(scenario, persona_context, state)
        history = self._format_history(state)
        
        parts = [user_message]
        if history:
            parts.append(history)
        
        full_message = "\n\n".join(parts)
        
        prompt = f"{system_prompt}\n\n{full_message}"
        
        logger.info(f"[{self.agent_name}] Gen {state.get('generation_count', 1)}: Querying LLM...")
        
        try:
            reaction = await robust_llm_call(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                structured_output=AgentReaction,
            )
            
            logger.info(
                f"[{self.agent_name}] ✅ Response: {reaction.reaction[:50]}... "
                f"→ {reaction.suggested_next_agent or 'no suggestion'} "
            )
            
            return reaction
        
        except Exception as e:
            logger.error(f"[{self.agent_name}] ❌ Failed after retries: {str(e)}")
            return

class EmotionalAgent(PersonaAgent):
    
    def __init__(self, model = None, temperature: float = 0.8):
        super().__init__(agent_name="emotional", model=model, temperature=temperature)
    
    def get_system_prompt(self) -> str:
        return AgentPrompts.emotional_system_prompt()
    
    def build_user_message(
        self,
        scenario: str,
        persona_context: Dict[str, Any],
        state: GraphState
    ) -> str:
        info = self._extract_persona_info(persona_context)
        ocean = self._extract_ocean(persona_context)
        news_ctx = self._extract_news_context(persona_context)
        return f"""СЦЕНАРИЙ:
{scenario}
{news_ctx}
ПРОФИЛЬ ПЕРСОНЫ:
- Возраст: {info['age_group']}
- Пол: {info['gender']}

ПСИХОЛОГИЧЕСКИЙ ПОРТРЕТ (как она себя ОЩУЩАЕТ):
- Открытость новому: {ocean['openness']}
- Добросовестность: {ocean['conscientiousness']}
- Общительность: {ocean['extraversion']}
- Сотрудничество: {ocean['agreeableness']}
- Эмоциональная стабильность: {ocean['neuroticism']}

Какие ПОДЛИННЫЕ ЭМОЦИИ испытывает эта персона?
Какой её инстинктивный отклик на этот сценарий?
Что она ЧУВСТВУЕТ, даже если не думает?"""


class RationalAgent(PersonaAgent):
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3):
        super().__init__(agent_name="rational", model=model, temperature=temperature)
    
    def get_system_prompt(self) -> str:
        return AgentPrompts.rational_system_prompt()
    
    def build_user_message(
        self,
        scenario: str,
        persona_context: Dict[str, Any],
        state: GraphState
    ) -> str:
        info = self._extract_persona_info(persona_context)
        ocean = self._extract_ocean(persona_context)
        news_ctx = self._extract_news_context(persona_context)
        return f"""СЦЕНАРИЙ:
{scenario}
{news_ctx}
ПРОФИЛЬ ПЕРСОНЫ:
- Образование: {info['education']}
- Доход: {info['income_level']}
- Возраст: {info['age_group']}
- Регион: {info['region']}
- Количество детей: {info['children_group']}
- Место работы: {info['occupation']}
- Семейное положение: {info['marital_status']}

АНАЛИТИЧЕСКИЕ ОСОБЕННОСТИ (как она ДУМАЕТ):
- Открытость к анализу: {ocean['openness']}
- Организованность мышления: {ocean['conscientiousness']}

Какова ЛОГИКА её выбора?
Какие ФАКТОРЫ и ПОСЛЕДСТВИЯ она видит?
Какой РИСК и какой ВЫИГРЫШ?
Что выбрала бы её ЛОГИКА?"""


class SocialAgent(PersonaAgent):
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.6):
        super().__init__(agent_name="social", model=model, temperature=temperature)
    
    def get_system_prompt(self) -> str:
        return AgentPrompts.social_system_prompt()
    
    def build_user_message(
        self,
        scenario: str,
        persona_context: Dict[str, Any],
        state: GraphState
    ) -> str:
        info = self._extract_persona_info(persona_context)
        ocean = self._extract_ocean(persona_context)
        news_ctx = self._extract_news_context(persona_context)
        return f"""СЦЕНАРИЙ:
{scenario}
{news_ctx}
ПРОФИЛЬ ПЕРСОНЫ:
- Образование: {info['education']}
- Пол: {info['gender']}
- Возраст: {info['age_group']}
- Семейное положение: {info['marital_status']}
- Регион: {info['region']}
- Количество детей: {info['children_group']}
- Место работы: {info['occupation']}
- Доход: {info['income_level']}

СОЦИАЛЬНЫЕ ОСОБЕННОСТИ (как она ВЗАИМОДЕЙСТВУЕТ):
- Общительность: {ocean['extraversion']}
- Сотрудничество: {ocean['agreeableness']}

Как это решение повлияет на её ОТНОШЕНИЯ?
Как её ОКРУЖЕНИЕ отреагирует?
Что это сделает с её СОЦИАЛЬНЫМ СТАТУСОМ и РЕПУТАЦИЕЙ?
Что выберет она ради БЛИЗКИХ?"""


class IdeologicalAgent(PersonaAgent):
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.5):
        super().__init__(agent_name="ideological", model=model, temperature=temperature)
    
    def get_system_prompt(self) -> str:
        return AgentPrompts.ideological_system_prompt()
    
    def build_user_message(
        self,
        scenario: str,
        persona_context: Dict[str, Any],
        state: GraphState
    ) -> str:
        info = self._extract_persona_info(persona_context)
        ocean = self._extract_ocean(persona_context)
        news_ctx = self._extract_news_context(persona_context)
        return f"""СЦЕНАРИЙ:
{scenario}
{news_ctx}
ПРОФИЛЬ ПЕРСОНЫ:
- Образование: {info['education']}
- Пол: {info['gender']}
- Возраст: {info['age_group']}
- Семейное положение: {info['marital_status']}
- Регион: {info['region']}
- Количество детей: {info['children_group']}
- Место работы: {info['occupation']}
- Доход: {info['income_level']}

ЦЕННОСТНЫЕ ОСОБЕННОСТИ (что она СЧИТАЕТ ПРАВИЛЬНЫМ):
- Открытость к пересмотру убеждений: {ocean['openness']}
- Следование принципам: {ocean['conscientiousness']}
- Альтруизм/Эгоизм: {ocean['agreeableness']}

Соответствует ли это решение её ЦЕННОСТЯМ?
Что это скажет о её ЛИЧНОСТИ и ИДЕНТИЧНОСТИ?
Останется ли она верна себе?
Какова МОРАЛЬНАЯ позиция?"""


class DecisionAgent(PersonaAgent):
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.2):
        super().__init__(agent_name="decision", model=model, temperature=temperature)
    
    def get_system_prompt(self) -> str:
        return AgentPrompts.decision_system_prompt()
    
    def build_user_message(
        self,
        scenario: str,
        persona_context: Dict[str, Any],
        state: GraphState
    ) -> str:
        info = self._extract_persona_info(persona_context)
        
        all_voices = []
        for agent_type in ["emotional", "rational", "social", "ideological"]:
            history = state.get(f"{agent_type}_history", [])
            if history:
                all_voices.append(f"\n{agent_type.upper()} VOICE:")
                for entry in history:
                    all_voices.append(f"- {entry.get('reaction', 'N/A')}")
        
        voices_text = "\n".join(all_voices) if all_voices else "No voices yet"
        news_ctx = self._extract_news_context(persona_context)
        return f"""СЦЕНАРИЙ:
    {scenario}
    {news_ctx}
    ВСЕ ГОЛОСА СОЗНАНИЯ:
    {voices_text}

    ПРОФИЛЬ ПЕРСОНЫ:
    - Образование: {info['education']}
    - Пол: {info['gender']}
    - Возраст: {info['age_group']}
    - Семейное положение: {info['marital_status']}
    - Доход: {info['income_level']}

    ТВОЯ ЗАДАЧА:
    1. Синтезируй ЭМОЦИОНАЛЬНЫЙ, РАЦИОНАЛЬНЫЙ, СОЦИАЛЬНЫЙ и ИДЕОЛОГИЧЕСКИЙ голоса
    2. Найди КОНСЕНСУС или разреши КОНФЛИКТ
    3. Прими ФИНАЛЬНОЕ РЕШЕНИЕ: True (ДА) или False (НЕТ)
    4. Оцени УВЕРЕННОСТЬ (0.0-1.0)

    РЕШИ СЕЙЧАС."""
    
    async def run(
        self,
        scenario: str,
        persona_context: Dict[str, Any],
        state: GraphState
    ) -> DecisionOutput:
        """DECISION агент возвращает DecisionOutput, а не AgentReaction"""
        
        info = self._extract_persona_info(persona_context)
        
        system_prompt = self.get_system_prompt()
        user_message = self.build_user_message(scenario, persona_context, state)
        
        prompt = f"{system_prompt}\n\n{user_message}"
        
        logger.info(f"[DECISION] Synthesizing all perspectives...")
        
        try:
            result = await robust_llm_call(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                structured_output=DecisionOutput,  
            )
            
            logger.info(f"[DECISION] Decision: {result.decision}, Confidence: {result.confidence}") 
            return result  
            
        except Exception as e:
            logger.error(f"[DECISION] Error: {e}")
            raise


class RoutingEngine:
    """Real routing logic based on agent suggestions"""
    
    def __init__(self):
        self.agent_to_candidates = {
            "emotional": ["rational", "social", "ideological"],
            "rational": ["emotional", "social", "ideological"],
            "social": ["emotional", "rational", "ideological"],
            "ideological": ["emotional", "rational", "social"],
        }
    
    def get_next_agent(
        self,
        current_agent: str,
        suggested_next: Optional[str],
        generation_count: int,
        max_generations: int,
    ) -> str:
        
        if generation_count >= max_generations:
            logger.info(f"🔴 Max generations ({max_generations}) reached → DECISION")
            return "decision"
        
        all_agents = ["emotional", "rational", "social", "ideological"]
        
        if (suggested_next and 
            suggested_next != current_agent and 
            suggested_next in all_agents):
            logger.info(f"✅ Agent {current_agent} → {suggested_next}")
            return suggested_next.lower()
        
        if current_agent in all_agents:
            current_idx = all_agents.index(current_agent)
            next_idx = (current_idx + 1) % len(all_agents)
            next_agent = all_agents[next_idx]
            logger.info(f"🔄 Agent {current_agent} → {next_agent} (cycle)")
            return next_agent
        
        return "decision"



class MultiAgentReasoner:
    
    def __init__(
        self,
        persona_context: Dict,
        world_context: Dict = None,
        max_generations: int = 5,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE
    ):
        self.max_generations = max_generations
        self.model = model
        self.world_context = world_context or {}
        # world_context передаётся агентам через persona_context["news_context"]
        self.persona_context = {**persona_context, "news_context": self.world_context}
        
        self.agents: Dict[str, PersonaAgent] = {
            "emotional": EmotionalAgent(model=model),
            "rational": RationalAgent(model=model),
            "social": SocialAgent(model=model),
            "ideological": IdeologicalAgent(model=model),
        }
        self.decision_agent = DecisionAgent(model=model)
        
        self.router = RoutingEngine()
        
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        
        for agent_name in self.agents.keys():
            graph.add_node(agent_name.lower(), self._run_agent_node)
        
        graph.add_node("decision", self._run_decision_node)
        
        graph.add_edge(START, "emotional")
        edge_map = {
            "emotional": "emotional",
            "rational": "rational",
            "social": "social",
            "ideological": "ideological",
            "decision": "decision"  
        }
        
        for agent_name in self.agents.keys():
            graph.add_conditional_edges(
                agent_name.lower(),
                self._route_from_agent,
                edge_map
            )
        
        graph.add_edge("decision", END)
        
        return graph.compile()

    async def _run_agent_node(self, state: GraphState) -> GraphState:
        """Generic node to run any reasoning agent"""
        agent_name = state["current_agent"]
        agent = self.agents[agent_name]
        
        reaction = await agent.run(
            state["scenario"],
            state["persona_context"],
            state
        )
        
        history_key = f"{agent_name.lower()}_history"
        current_history = state[history_key]
        current_history.append({
            "reasoning": reaction.reasoning,
            "reaction": reaction.reaction,
            "suggested_next_agent": reaction.suggested_next_agent
        })
        
        state["generation_count"] += 1
        
        next_agent = self.router.get_next_agent(
            current_agent=agent_name,
            suggested_next=reaction.suggested_next_agent,
            generation_count=state["generation_count"],
            max_generations=state["max_generations"],
        )
        
        state["current_agent"] = next_agent
        
        return state
    
    def _route_from_agent(self, state: GraphState) -> str:
        return state["current_agent"]

    
    async def _run_decision_node(self, state: GraphState) -> GraphState:
        decision_result = await self.decision_agent.run(
            state["scenario"],  
            state["persona_context"],  
            state 
        )
        state["final_decision"] = decision_result 
        return state

    async def run(
        self,
        scenario: str,
        max_generations: Optional[int] = None,
        persona_id: Optional[str] = None
    ) -> Dict:
        
        effective_max_gen = max_generations or self.max_generations
        
        logger.info(f"🚀 Starting reasoning for persona: {persona_id} on scenario: '{scenario}'")
        
        initial_state = GraphState(
            persona_id=persona_id,
            scenario=scenario,
            persona_context=self.persona_context,
            max_generations=effective_max_gen,
            generation_count=0,
            current_agent="emotional", 
            emotional_history=[],
            rational_history=[],
            social_history=[],
            ideological_history=[],
        )
        
        final_state = await self.graph.ainvoke(initial_state)
        
        logger.info(f"✅ Reasoning complete for {persona_id}")
        
        result = {
            "persona_id": persona_id,
            "scenario": scenario,
            "persona_context": self.persona_context,
            "emotional_history": final_state.get("emotional_history", []),
            "rational_history": final_state.get("rational_history", []),
            "social_history": final_state.get("social_history", []),
            "ideological_history": final_state.get("ideological_history", []),
            "generation_count": final_state.get("generation_count", 0),
            "max_generations": final_state.get("max_generations", 0),
            "final_decision": None,  # Будет преобразовано ниже
            "timestamp": datetime.utcnow().isoformat()
        }
        
        final_decision = final_state.get("final_decision")
        if final_decision is not None:
            if hasattr(final_decision, "dict"):  
                result["final_decision"] = final_decision.dict()
            elif isinstance(final_decision, dict): 
                result["final_decision"] = final_decision
            else:  
                try:
                    result["final_decision"] = dict(final_decision)
                except:
                    result["final_decision"] = str(final_decision)
        
        return result

    
    async def answer_survey_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Запускает мультиагента для каждого вопроса и возвращает полные результаты
        """
        all_results = []
        
        for i, question in enumerate(questions):
            logger.info(f"[SURVEY] Запуск вопроса {i+1}/{len(questions)}: {question[:50]}...")
            
            question_result = await self.run(
                scenario=question,
                max_generations=self.max_generations,
                persona_id=self.persona_context.get("name", f"persona_{i}")
            )
            
            all_results.append({
                "question": question,
                "question_index": i,
                "scenario": question,
                "full_state": question_result,  
                "timestamp": datetime.utcnow().isoformat()
            })
            
        return all_results


async def main():
    """Главная функция для CLI"""
    
    parser = argparse.ArgumentParser(
        description='Survey Pipeline with Multi-Agent Simulation'
    )
    
    parser.add_argument(
        '--persona',
        type=lambda x: json.load(open(x)),
        required=True,
        help='Путь до JSON файла с описанием персоны (обязательно)'
    )
    
    parser.add_argument(
        '--question',
        type=str,
        default=None,
        help='Один вопрос для обработки (опционально)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='answer.json',
        help='Путь для сохранения результатов (по умолчанию: survey_results.json)'
    )
    
    parser.add_argument(
        '--max-generations',
        type=int,
        default=5,
        help='Максимальное количество итераций агентов (по умолчанию: 5)'
    )
    
    args = parser.parse_args()
    
    try:
        # Инициализируем runner
        logger.info("🚀 Инициализация Multi-Agent Survey Runner...")
        runner = MultiAgentReasoner(args.persona)
        questions = [args.question]
        
        results = await runner.answer_survey_questions(questions)
        
        # Сохраняем результаты
        runner.save_results(results, args.output)
        
        # Выводим финальную статистику
        logger.info("\n" + "="*70)
        logger.info("📊 ИТОГОВАЯ СТАТИСТИКА")
        logger.info("="*70)
        logger.info(f"Всего вопросов обработано: {len(results)}")
        
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        logger.info(f"✓ Успешно: {len(successful)}")
        logger.info(f"✗ Ошибок: {len(failed)}")
        
        if successful:
            yes_count = sum(1 for r in successful if r.get('decision') is True)
            logger.info(f"Ответы 'ДА': {yes_count}")
            logger.info(f"Ответы 'НЕТ': {len(successful) - yes_count}")
        
        logger.info("="*70 + "\n")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ Прервано пользователем")
        return 130


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)