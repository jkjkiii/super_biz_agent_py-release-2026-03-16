# test_routing.py
import asyncio
from app.core.intent_router import intent_router, Intent
from app.core.route_handlers import route_handlers

async def test():
    cases = [
        ("你好", "chat"),
        ("什么是 Kubernetes Pod？", "knowledge"),
        ("CPU 一直很高怎么办？", "troubleshoot"),
        ("帮我重启 nginx", "control"),
    ]
    
    for q, expected in cases:
        route = intent_router.route(q)
        print(f"\n[{expected}] {q}")
        print(f"意图: {route.intent.value}, 置信度: {route.confidence:.2f}")
        
        if route.intent == Intent.CHAT:
            print("回答:", (await route_handlers.handle_chat(q))[:100])
        elif route.intent == Intent.KNOWLEDGE:
            print("回答:", (await route_handlers.handle_knowledge(q))[:100])
        elif route.intent == Intent.TROUBLESHOOT:
            print("回答:", (await route_handlers.handle_troubleshoot(q, "test"))[:100])
        elif route.intent == Intent.CONTROL:
            print("回答:", (await route_handlers.handle_control(q, "test"))[:100])

asyncio.run(test())