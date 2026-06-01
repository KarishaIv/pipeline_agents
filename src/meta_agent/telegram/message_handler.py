"""Message handlers and output sending for Telegram bot."""

import logging
import asyncio
from typing import Any


from src.meta_agent.telegram.update_parser import TelegramMessage, parse_message_text
from src.meta_agent.telegram.bot_client import TelegramBotClient
from src.meta_agent.telegram.meta_agent_client import MetaAgentClient
from src.meta_agent.telegram.session_store import TelegramSessionStore, Session

logger = logging.getLogger(__name__)


def is_default_session_name(name: str) -> bool:
    """Check if session name is a default auto-generated name.

    Default names follow the pattern 'session_<number>'.

    Args:
        name: Session name to check.

    Returns:
        True if the name is a default session name, False otherwise.
    """
    if not name.startswith("session_"):
        return False
    try:
        int(name[8:])  # Try to parse the number after "session_"
        return True
    except ValueError:
        return False


def should_delete_session(session: Session) -> bool:
    """Check if a session should be deleted when switching/creating new session.

    Sessions are deleted only if they:
    - Have a default name (session_<number>)
    - AND have no messages

    Args:
        session: Session object to check.

    Returns:
        True if the session should be deleted, False otherwise.
    """
    return is_default_session_name(session.name) and not session.has_messages


def next_default_session_name(sessions: list) -> str:
    """Generate next unused default session name (session_N).

    Uses the highest number among existing default-named sessions + 1.
    Handles gaps (e.g. session_1, session_3 → next is session_4).
    Falls back to session_1 if no defaults exist.

    Args:
        sessions: List of existing Session objects for the user.

    Returns:
        Next default name like 'session_4'.
    """
    max_n = 0
    for session in sessions:
        if is_default_session_name(session.name):
            try:
                n = int(session.name[8:])
                if n > max_n:
                    max_n = n
            except ValueError:
                continue
    return f"session_{max_n + 1}"


def get_commands_keyboard() -> dict[str, Any]:
    """Get the reply keyboard markup for bot commands.

    Returns:
        ReplyKeyboardMarkup dict with command buttons.
    """
    return {
        "keyboard": [
            [{"text": "/help"}],
            [{"text": "/new"}, {"text": "/sessions"}],
            [{"text": "/switch"}, {"text": "/delete"}],
        ],
        "resize_keyboard": True,
        "is_persistent": True,
        "one_time_keyboard": False,
    }


def get_session_switch_keyboard(sessions: list) -> dict[str, Any]:
    """Get inline keyboard for switching sessions.

    Args:
        sessions: List of session objects.

    Returns:
        InlineKeyboardMarkup dict with session buttons.
    """
    buttons = []
    for session in sessions:
        if session.is_active:
            marker = "🟢"
            button_text = f"{marker} {session.name} (active)"
            buttons.append(
                [{"text": button_text, "callback_data": "switch_active"}]
            )
        else:
            marker = "⚪️"
            button_text = f"{marker} {session.name}"
            buttons.append(
                [{"text": button_text, "callback_data": f"switch:{session.name}"}]
            )

    return {"inline_keyboard": buttons}


def get_session_delete_keyboard(sessions: list) -> dict[str, Any]:
    """Get inline keyboard for deleting sessions.

    Args:
        sessions: List of session objects.

    Returns:
        InlineKeyboardMarkup dict with session buttons.
    """
    buttons = []

    for session in sessions:
        if session.is_active:
            marker = "🟢"
            button_text = f"{marker} {session.name} (active)"
            buttons.append(
                [{"text": button_text, "callback_data": "delete_active"}]
            )
        else:
            marker = "⚪️"
            button_text = f"{marker} {session.name}"
            buttons.append(
                [{"text": button_text, "callback_data": f"delete:{session.name}"}]
            )

    return {"inline_keyboard": buttons}


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

    async def _safe_send_message(self, chat_id: int, text: str, **kwargs: Any) -> bool:
        """Send a message without letting Telegram network errors break the flow."""
        try:
            await self.telegram.send_message(chat_id, text, **kwargs)
            return True
        except Exception as exc:
            logger.error("Failed to send Telegram message to chat %s: %s", chat_id, exc)
            return False

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
                reply_markup=get_commands_keyboard(),
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
                reply_markup=get_commands_keyboard(),
            )
        elif command == "new":
            session_name = text.strip() if text.strip() else None

            # Generate default name if not provided
            if not session_name:
                sessions = self.session_store.list_sessions(msg.chat_id, msg.user_id)
                session_name = next_default_session_name(sessions)

            # Get current active session before creating new one
            current_session = self.session_store.get_active_session(msg.chat_id, msg.user_id)

            # Create new session with generated thread_id
            session = self.session_store.create_session(
                msg.chat_id, msg.user_id, thread_id="", name=session_name, activate=True
            )

            # Delete the old session if it should be deleted
            if current_session and should_delete_session(current_session):
                self.session_store.delete_session(msg.chat_id, msg.user_id, current_session.name)

            await self.telegram.send_message(
                msg.chat_id,
                f"✨ New session '<b>{session.name}</b>' created and activated!\n"
                f"Ask me something to start the conversation.",
                reply_markup=get_commands_keyboard(),
            )
        elif command == "sessions":
            sessions = self.session_store.list_sessions(msg.chat_id, msg.user_id)

            if not sessions:
                await self.telegram.send_message(
                    msg.chat_id,
                    "You don't have any sessions yet. Use /new to create one.",
                    reply_markup=get_commands_keyboard(),
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
            await self.telegram.send_message(msg.chat_id, "\n".join(lines), reply_markup=get_commands_keyboard())

        elif command == "switch":
            if not text.strip():
                sessions = self.session_store.list_sessions(msg.chat_id, msg.user_id)

                if not sessions:
                    await self.telegram.send_message(
                        msg.chat_id,
                        "You don't have any sessions yet. Use /new to create one.",
                        reply_markup=get_commands_keyboard(),
                    )
                    return

                keyboard = get_session_switch_keyboard(sessions)
                await self.telegram.send_message(
                    msg.chat_id,
                    "📋 Select a session to switch to:",
                    reply_markup=keyboard,
                )
                return

            session_name = text.strip()

            # Get current active session before switching
            current_session = self.session_store.get_active_session(msg.chat_id, msg.user_id)

            # Switch to new session
            session = self.session_store.switch_session(msg.chat_id, msg.user_id, session_name)

            if session:
                # Delete the old session if it should be deleted
                if current_session and should_delete_session(current_session):
                    self.session_store.delete_session(msg.chat_id, msg.user_id, current_session.name)

                await self.telegram.send_message(
                    msg.chat_id,
                    f"✅ Switched to session '<b>{session.name}</b>'\n"
                    f"Continue your conversation from where you left off.",
                    reply_markup=get_commands_keyboard(),
                )
            else:
                await self.telegram.send_message(
                    msg.chat_id,
                    f"❌ Session '<b>{session_name}</b>' not found.\n"
                    f"Use /sessions to see available sessions.",
                    reply_markup=get_commands_keyboard(),
                )

        elif command == "delete":
            if not text.strip():
                sessions = self.session_store.list_sessions(msg.chat_id, msg.user_id)

                if not sessions:
                    await self.telegram.send_message(
                        msg.chat_id,
                        "You don't have any sessions yet. Use /new to create one.",
                        reply_markup=get_commands_keyboard(),
                    )
                    return

                keyboard = get_session_delete_keyboard(sessions)
                await self.telegram.send_message(
                    msg.chat_id,
                    "🗑️ Select a session to delete:",
                    reply_markup=keyboard,
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
                    reply_markup=get_commands_keyboard(),
                )
                return
            
            deleted = self.session_store.delete_session(msg.chat_id, msg.user_id, session_name)
            if deleted:
                await self.telegram.send_message(
                    msg.chat_id,
                    f"✅ Session '<b>{session_name}</b>' deleted.",
                    reply_markup=get_commands_keyboard(),
                )
            else:
                await self.telegram.send_message(
                    msg.chat_id,
                    f"❌ Session '<b>{session_name}</b>' not found.",
                    reply_markup=get_commands_keyboard(),
                )
        else:
            await self.telegram.send_message(
                msg.chat_id, f"❌ Unknown command: /{command}", reply_markup=get_commands_keyboard()
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
                    # No session exists, create default one with generated thread_id
                    session = self.session_store.create_session(
                        msg.chat_id, msg.user_id, thread_id="", name="default", activate=True
                    )
                    logger.info("Created default session for chat %d, user %d", msg.chat_id, msg.user_id)

                thread_id = session.thread_id

                logger.info(
                    "Processing question from chat %d, session '%s' (thread %s): %s",
                    msg.chat_id,
                    session.name,
                    thread_id,
                    question[:100],
                )

                response = await self.meta_agent.ask(question, thread_id)

                # Mark session as having messages after successful meta-agent response
                self.session_store.mark_session_has_messages(msg.chat_id, msg.user_id, thread_id)

                await self.send_outputs(msg.chat_id, response.outputs)

            except Exception as e:
                logger.exception("Error handling question in chat %d", msg.chat_id)
                await self._safe_send_message(
                    msg.chat_id,
                    f"❌ Error: {str(e)[:200]}\n\nPlease try again or contact support.",
                )

    async def send_outputs(self, chat_id: int, outputs: list) -> None:
        """Send all outputs from meta-agent to Telegram, dispatching by type.

        Args:
            chat_id: Telegram chat ID.
            outputs: List of AgentOutput objects (TextOutput, JsonOutput, ImageOutput, FileOutput).
        """
        for output in outputs:
            output_type = getattr(output, "type", None)

            if output_type == "text":
                chunks = parse_message_text(output.text)
                for chunk in chunks:
                    await self._safe_send_message(chat_id, chunk)

            elif output_type == "json":
                import json
                json_str = json.dumps(output.data, indent=2, ensure_ascii=False)
                caption = output.caption or "Data"
                message_text = f"<b>{caption}</b>\n\n<pre>{json_str}</pre>"
                chunks = parse_message_text(message_text)
                for chunk in chunks:
                    await self._safe_send_message(chat_id, chunk)

            elif output_type == "image":
                caption = output.caption or None
                try:
                    # Fetch artifact bytes from the API
                    content, mime_type, filename = await self.meta_agent.fetch_artifact_bytes(output.url)

                    # Upload via multipart
                    await self.telegram.send_photo(
                        chat_id,
                        content=content,
                        filename=filename,
                        mime_type=mime_type,
                        caption=caption
                    )
                except Exception as e:
                    logger.error("Failed to send photo: %s", e)
                    await self._safe_send_message(
                        chat_id,
                        f"⚠️ Could not send image: {str(e)[:100]}",
                    )

            elif output_type == "file":
                caption = output.caption or output.filename
                try:
                    content, mime_type, filename = await self.meta_agent.fetch_artifact_bytes(
                        output.download_url
                    )
                    await self.telegram.send_document(
                        chat_id,
                        content=content,
                        filename=filename or output.filename,
                        mime_type=mime_type or output.mime_type,
                        caption=caption,
                    )
                except Exception as e:
                    logger.error("Failed to send document: %s", e)
                    await self._safe_send_message(
                        chat_id,
                        f"⚠️ Could not send file ({output.filename}): {str(e)[:100]}",
                    )

            else:
                # Fallback for unknown types
                logger.warning("Unknown output type: %s", output_type)
                await self._safe_send_message(
                    chat_id, f"⚠️ Unknown output type: {output_type}"
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

    async def handle_callback_query(self, callback_id: str, chat_id: int, user_id: int, data: str) -> None:
        """Handle a callback query from inline keyboard.

        Args:
            callback_id: Callback query ID to acknowledge.
            chat_id: Telegram chat ID.
            user_id: Telegram user ID.
            data: Callback data string.
        """
        try:
            if data == "switch_active":
                await self.telegram.answer_callback_query(
                    callback_id,
                    text="This session is already active",
                    show_alert=False,
                )

            elif data.startswith("switch:"):
                session_name = data[7:]  # Remove "switch:" prefix

                # Get current active session before switching
                current_session = self.session_store.get_active_session(chat_id, user_id)

                # Switch to new session
                session = self.session_store.switch_session(chat_id, user_id, session_name)

                if session:
                    # Delete the old session if it should be deleted
                    if current_session and should_delete_session(current_session):
                        self.session_store.delete_session(chat_id, user_id, current_session.name)

                    await self.telegram.answer_callback_query(
                        callback_id,
                        text=f"Switched to '{session_name}'",
                    )
                    await self.telegram.send_message(
                        chat_id,
                        f"✅ Switched to session '<b>{session.name}</b>'\n"
                        f"Continue your conversation from where you left off.",
                        reply_markup=get_commands_keyboard(),
                    )
                else:
                    await self.telegram.answer_callback_query(
                        callback_id,
                        text="Session not found",
                        show_alert=True,
                    )

            elif data == "delete_active":
                await self.telegram.answer_callback_query(
                    callback_id,
                    text="Cannot delete the active session. Switch to another session first.",
                    show_alert=True,
                )

            elif data.startswith("delete:"):
                session_name = data[7:]  # Remove "delete:" prefix

                deleted = self.session_store.delete_session(chat_id, user_id, session_name)
                if deleted:
                    await self.telegram.answer_callback_query(
                        callback_id,
                        text=f"Session '{session_name}' deleted",
                    )
                    await self.telegram.send_message(
                        chat_id,
                        f"✅ Session '<b>{session_name}</b>' deleted.",
                        reply_markup=get_commands_keyboard(),
                    )
                else:
                    await self.telegram.answer_callback_query(
                        callback_id,
                        text="Session not found",
                        show_alert=True,
                    )
            else:
                logger.warning("Unknown callback data: %s", data)
                await self.telegram.answer_callback_query(callback_id, text="Unknown action")

        except Exception as e:
            logger.exception("Error handling callback query in chat %d", chat_id)
            await self.telegram.answer_callback_query(
                callback_id,
                text=f"Error: {str(e)[:50]}",
                show_alert=True,
            )
