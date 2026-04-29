"""Message handlers and output sending for Telegram bot."""

import logging
import asyncio

from meta_agent import TextOutput, JsonOutput, FileOutput

from src.meta_agent.telegram.update_parser import TelegramMessage, parse_message_text
from src.meta_agent.telegram.bot_client import TelegramBotClient
from src.meta_agent.telegram.meta_agent_client import MetaAgentClient
from src.meta_agent.telegram.session_store import TelegramSessionStore

logger = logging.getLogger(__name__)


class MessageHandler:
    """Handles incoming Telegram messages and coordinates responses."""

    def __init__(
        self,
        telegram_client: TelegramBotClient,
        meta_agent_client: MetaAgentClient,
        session_store: TelegramSessionStore,
    ):
        """Initialize handler.

        Args:
            telegram_client: Telegram Bot API client.
            meta_agent_client: Meta-agent HTTP client.
            session_store: Session storage.
        """
        self.telegram = telegram_client
        self.meta_agent = meta_agent_client
        self.session_store = session_store
        self.per_chat_locks = {}

    def _get_chat_lock(self, chat_id: int) -> asyncio.Lock:
        """Get or create a lock for a chat to ensure ordered processing."""
        if chat_id not in self.per_chat_locks:
            self.per_chat_locks[chat_id] = asyncio.Lock()
        return self.per_chat_locks[chat_id]

    async def handle_command(
        self, msg: TelegramMessage, command: str, text: str
    ) -> None:
        """Handle a command message.

        Args:
            msg: Parsed message.
            command: Command name (without /).
            text: Optional command arguments.
        """
        if command == "start":
            await self.telegram.send_message(
                msg.chat_id,
                "👋 Welcome to Meta Agent Bot!\n\n"
                "Send me a question about your pipeline data and I'll analyze it for you.\n\n"
                "Commands:\n"
                "/help — Show help\n"
                "/new [name] — Start a new session\n"
                "/sessions — List your sessions\n"
                "/switch <name> — Switch to a different session\n"
                "/delete <name> — Delete a session",
                parse_mode=None,
            )
        elif command == "help":
            await self.telegram.send_message(
                msg.chat_id,
                "📚 Help\n\n"
                "Send any question about your pipeline and I'll provide analysis.\n\n"
                "<b>Session Management:</b>\n"
                "• Each conversation is saved in a session\n"
                "• You can have multiple sessions and switch between them\n"
                "• Sessions preserve your conversation history\n\n"
                "<b>Commands:</b>\n"
                "/new [name] — Create new session (default name: 'session_N')\n"
                "/sessions — List all your sessions\n"
                "/switch &lt;name&gt; — Switch to another session\n"
                "/delete &lt;name&gt; — Delete a session\n"
                "/help — Show this help",
            )
        elif command == "new":
            session_name = text.strip() if text.strip() else None
            
            # Generate default name if not provided
            if not session_name:
                sessions = self.session_store.list_sessions(msg.chat_id, msg.user_id)
                session_name = f"session_{len(sessions) + 1}"
            
            # Create new session with thread_id=-1 to signal new conversation
            session = self.session_store.create_session(
                msg.chat_id, msg.user_id, thread_id="-1", name=session_name, activate=True
            )
            await self.telegram.send_message(
                msg.chat_id,
                f"✨ New session '<b>{session.name}</b>' created and activated!\n"
                f"Ask me something to start the conversation.",
            )
        elif command == "sessions":
            sessions = self.session_store.list_sessions(msg.chat_id, msg.user_id)
            
            if not sessions:
                await self.telegram.send_message(
                    msg.chat_id,
                    "You don't have any sessions yet. Use /new to create one.",
                )
                return
            
            lines = ["📋 <b>Your Sessions:</b>\n"]
            for session in sessions:
                active_marker = "🟢 " if session.is_active else "⚪️ "
                updated = session.updated_at.strftime("%Y-%m-%d %H:%M")
                lines.append(
                    f"{active_marker}<b>{session.name}</b> (updated: {updated})"
                )
            
            lines.append("\nUse /switch &lt;name&gt; to switch sessions")
            await self.telegram.send_message(msg.chat_id, "\n".join(lines))
            
        elif command == "switch":
            if not text.strip():
                await self.telegram.send_message(
                    msg.chat_id,
                    "❌ Please specify a session name: /switch &lt;name&gt;\n"
                    "Use /sessions to see available sessions.",
                )
                return
            
            session_name = text.strip()
            session = self.session_store.switch_session(msg.chat_id, msg.user_id, session_name)
            
            if session:
                await self.telegram.send_message(
                    msg.chat_id,
                    f"✅ Switched to session '<b>{session.name}</b>'\n"
                    f"Continue your conversation from where you left off.",
                )
            else:
                await self.telegram.send_message(
                    msg.chat_id,
                    f"❌ Session '<b>{session_name}</b>' not found.\n"
                    f"Use /sessions to see available sessions.",
                )
        
        elif command == "delete":
            if not text.strip():
                await self.telegram.send_message(
                    msg.chat_id,
                    "❌ Please specify a session name: /delete &lt;name&gt;\n"
                    "Use /sessions to see available sessions.",
                )
                return
            
            session_name = text.strip()
            
            # Check if trying to delete active session
            active = self.session_store.get_active_session(msg.chat_id, msg.user_id)
            if active and active.name == session_name:
                await self.telegram.send_message(
                    msg.chat_id,
                    f"❌ Cannot delete active session '<b>{session_name}</b>'.\n"
                    f"Switch to another session first, or create a new one with /new",
                )
                return
            
            deleted = self.session_store.delete_session(msg.chat_id, msg.user_id, session_name)
            if deleted:
                await self.telegram.send_message(
                    msg.chat_id,
                    f"✅ Session '<b>{session_name}</b>' deleted.",
                )
            else:
                await self.telegram.send_message(
                    msg.chat_id,
                    f"❌ Session '<b>{session_name}</b>' not found.",
                )
        else:
            await self.telegram.send_message(
                msg.chat_id, f"❌ Unknown command: /{command}"
            )

    async def handle_question(self, msg: TelegramMessage) -> None:
        """Handle a regular question message.

        Args:
            msg: Parsed message with question text.
        """
        question = msg.text.strip()
        if not question:
            await self.telegram.send_message(
                msg.chat_id,
                "Please provide a question. Type /help for usage information.",
            )
            return

        lock = self._get_chat_lock(msg.chat_id)
        async with lock:
            try:
                await self.telegram.send_chat_action(msg.chat_id, "typing")

                # Get or create active session
                session = self.session_store.get_active_session(msg.chat_id, msg.user_id)
                
                if not session:
                    # No session exists, create default one with placeholder
                    session = self.session_store.create_session(
                        msg.chat_id, msg.user_id, thread_id="-1", name="default", activate=True
                    )
                    logger.info("Created default session for chat %d, user %d", msg.chat_id, msg.user_id)
                
                thread_id = session.thread_id if session.thread_id != "-1" else None
                
                logger.info(
                    "Processing question from chat %d, session '%s' (thread %s): %s",
                    msg.chat_id,
                    session.name,
                    thread_id,
                    question[:100],
                )

                response = await self.meta_agent.ask(question, thread_id)

                # If we used a placeholder, replace it with the real thread_id
                if session.thread_id == "-1":
                    self.session_store.replace_thread_id(
                        msg.chat_id, msg.user_id, "-1", response.thread_id
                    )
                    logger.info(
                        "Updated session '%s' with real thread ID: %s",
                        session.name,
                        response.thread_id
                    )

                await self.send_outputs(msg.chat_id, response.outputs)

            except Exception as e:
                logger.exception("Error handling question in chat %d", msg.chat_id)
                await self.telegram.send_message(
                    msg.chat_id,
                    f"❌ Error: {str(e)[:200]}\n\nPlease try again or contact support.",
                )

    async def send_outputs(self, chat_id: int, outputs: list) -> None:
        """Send all outputs from meta-agent to Telegram.

        Args:
            chat_id: Telegram chat ID.
            outputs: List of AgentOutput objects.
        """
        for output in outputs:
            if isinstance(output, TextOutput):
                chunks = parse_message_text(output.text)
                for chunk in chunks:
                    await self.telegram.send_message(chat_id, chunk)

            elif isinstance(output, JsonOutput):
                import json

                json_str = json.dumps(output.data, indent=2, ensure_ascii=False)
                caption = output.caption or "Data"
                message_text = f"<b>{caption}</b>\n\n<pre>{json_str}</pre>"
                chunks = parse_message_text(message_text)
                for chunk in chunks:
                    await self.telegram.send_message(chat_id, chunk)

            elif isinstance(output, FileOutput):
                caption = output.caption or output.filename
                try:
                    await self.telegram.send_document(
                        chat_id, output.download_url, caption=caption
                    )
                except Exception as e:
                    logger.error("Failed to send document: %s", e)
                    await self.telegram.send_message(
                        chat_id,
                        f"⚠️ Could not send document ({output.filename}): {str(e)[:100]}",
                    )

    async def handle_message(self, msg: TelegramMessage) -> None:
        """Route message to appropriate handler.

        Args:
            msg: Parsed Telegram message.
        """
        if msg.is_command and msg.command:
            await self.handle_command(msg, msg.command, msg.text)
        elif not msg.is_command and msg.text:
            await self.handle_question(msg)
