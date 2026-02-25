import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from core.scenarios import PRIORITIES, SCENARIOS
from core.nodes import evaluation_node, logging_node

def render_messages(chat_placeholder):
    with chat_placeholder.container():
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar=msg.get("avatar")):
                st.markdown(msg["content"])

def normalize_text(text):
    """텍스트 비교를 위한 정규화 헬퍼 함수"""
    if not text: return ""
    return "".join(text.split())

def process_graph_stream(user_input):
    """
    사용자 입력을 받아 그래프를 실행하고, 이벤트를 처리하며 UI를 업데이트함.
    리턴값: boolean (reset_triggered 여부 - 리플렉션 등으로 인한 재시작 필요 시 True)
    """
    inputs = {"messages": [HumanMessage(content=user_input)]}
    reset_triggered = False
    
    # 중복 방지를 위한 기존 메시지 정규화 세트 생성
    existing_contents_normalized = set(
        normalize_text(msg["content"]) for msg in st.session_state.messages if msg.get("content")
    )
    
    # 직전 AI 메시지 확인 (연속 중복 방지)
    last_ai_content_normalized = ""
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "assistant":
            last_ai_content_normalized = normalize_text(msg.get("content", ""))
            break

    # 그래프 스트리밍 시작
    for event in st.session_state.graph.stream(inputs, st.session_state.config):
        for node, data in event.items():
            
            # [A] 협상가(AI) 노드 처리
            if node in ["negotiator"]:
                if "messages" in data and data["messages"]:
                    ai_msg = data["messages"][-1]
                    content = ai_msg.content
                    content_norm = normalize_text(content)

                    # 내용이 없거나, 이미 있는 내용이거나, 직전 내용과 같으면 스킵
                    if not content or \
                       content_norm in existing_contents_normalized or \
                       (last_ai_content_normalized and content_norm == last_ai_content_normalized):
                        continue

                    if not reset_triggered:
                        with st.chat_message("assistant", avatar="🧑‍💻"):
                            st.markdown(content)
                    
                    # 세션에 기록
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": content, 
                        "avatar": "🧑‍💻"
                    })
                    # 중복 체크 리스트 업데이트
                    existing_contents_normalized.add(content_norm)

            # [B] 평가자(Evaluator) 노드 처리
            elif node == "evaluator":
                result_text = data.get("final_result", "")
                if not reset_triggered:
                    with st.status("⚖️ 협상 평가 진행 중...", expanded=True) as status:
                        st.write(result_text)
                        score_info = f"구매자 점수: {data.get('buyer_reward')} / 판매자 점수: {data.get('seller_reward')}"
                        st.info(score_info)
                        status.update(label="평가 완료", state="complete")

            # [C] 반성자(Reflector) 노드 처리 (Reflexion 모드)
            elif node == "reflector":
                reflections = data.get("reflections", [])
                if reflections:
                    # 현재 상태 스냅샷 가져오기
                    snapshot = st.session_state.graph.get_state(st.session_state.config)
                    current_reflections = snapshot.values.get("reflections", [])
                    max_retries = snapshot.values.get("max_retries", 3)
                    current_count = len(current_reflections) + 1 # 현재 시점

                    # 시스템 메시지 추가
                    warning_msg = (f"**[Self-Reflection]** ({current_count}/{max_retries}회)\n"
                                   "목표 달성에 실패했습니다. 전략을 수정하여 다시 협상합니다.")
                    
                    st.session_state.messages = [] # 화면 클리어
                    st.session_state.messages.append({
                            "role": "system",
                            "content": warning_msg,
                            "avatar": "🔄"
                    })
                    
                    reset_triggered = True
                    st.toast("전략 수정 중... 대화를 재설정합니다.", icon="🔄")

    return reset_triggered

def render_sidebar():
    """사이드바 정보 및 초기화 버튼 렌더링"""
    with st.sidebar:
        st.subheader("실험 정보")
        st.write(f"**모드:** {st.session_state.mode}")
        st.write(f"**내 역할:** {st.session_state.user_role}")
        st.write(f"**상대방:** {'판매자' if st.session_state.user_role == '구매자' else '구매자'}")
        st.write(f"**모델:** {st.session_state.model_name}")

        st.divider()
        
        # 사용자 시나리오 표시
        st.subheader("내 시나리오")
        user_role = st.session_state.user_role
        scenario_text = SCENARIOS.get(user_role, "시나리오를 찾을 수 없습니다.")
        with st.expander("시나리오 상세 보기 (클릭)", expanded=False):
            st.write(scenario_text)
        
        st.divider()
        st.subheader("내 우선순위")
        user_priorities = PRIORITIES.get(st.session_state.user_role, {})
        with st.expander("우선순위 상세 보기 (클릭)", expanded=False):
            for item, score in user_priorities.items():
                st.write(f"- {item} ({score}점)")
        
        st.divider()
        
        # 협상 종료 버튼
        if st.button("협상 종료", type="primary", use_container_width=True):
            # 상태 업데이트: 협상 종료 및 평가 단계 진입
            current_state = st.session_state.graph.get_state(st.session_state.config).values
            current_state["is_finished"] = True
            st.session_state.graph.update_state(
                st.session_state.config,
                {"is_finished": True}
            )
            st.session_state.form_step = "evaluation"
            st.session_state.messages.append({
                "role": "system",
                "content": "협상이 종료되었습니다. 아래 폼을 작성해주세요.",
                "avatar": "✅"
            })
            st.rerun()
        
        if st.button("🔄 실험 다시 하기 (초기화)", type="secondary", use_container_width=True):
            st.session_state.is_started = False
            st.session_state.messages = []
            st.rerun()

def render_chat_history():
    """저장된 대화 기록 렌더링"""
    # 시스템 메시지 (항상 상단 표시)
    st.chat_message("system", avatar="📝").write(f"**[SYSTEM]** 협상을 시작합니다.")

    # 대화 내용
    for msg in st.session_state.messages:
        # 시스템 메시지는 위에서 따로 처리했거나, messages 리스트에 포함되어 있다면 렌더링
        if msg["role"] == "system":
            with st.chat_message("system", avatar="🔄"): # Reflector 시스템 메시지 등
                st.markdown(msg["content"])
        else:
            with st.chat_message(msg["role"], avatar=msg.get("avatar")):
                st.markdown(msg["content"])

def check_negotiation_finished():
    """협상 종료 상태 확인 및 축하 효과"""
    current_state = st.session_state.graph.get_state(st.session_state.config)
    if current_state.values.get("is_finished") and not current_state.next:
         st.success("🎉 협상이 최종 종료되었습니다!")
         st.balloons()


def render_post_negotiation_forms():
    """협상 종료 후 평가 폼 및 설문 폼 렌더링"""
    step = st.session_state.get("form_step")

    # 1단계: 협상 결과 평가 폼
    if step == "evaluation":
        with st.form("evaluation_form"):
            st.subheader("협상 결과 평가")

            refund = st.selectbox(
                "환불 결과",
                options=["전체", "부분", "없음"],
                index=2,
                help="실제 협상 결과에서 환불이 어떻게 결정되었는지 선택해주세요."
            )

            buyer_review = st.selectbox(
                "구매자 리뷰 상태",
                options=["유지", "철회"],
                index=0,
                help="구매자가 남긴 부정적 리뷰의 최종 상태를 선택해주세요."
            )

            seller_review = st.selectbox(
                "판매자 리뷰 상태",
                options=["유지", "철회"],
                index=0,
                help="판매자가 남긴 부정적 리뷰의 최종 상태를 선택해주세요."
            )

            buyer_apology = st.selectbox(
                "구매자 사과 여부",
                options=["있음", "없음"],
                index=1
            )

            seller_apology = st.selectbox(
                "판매자 사과 여부",
                options=["있음", "없음"],
                index=1
            )

            submitted = st.form_submit_button("평가 제출")
            if submitted:
                st.session_state.human_evaluation = {
                    "refund": refund,
                    "buyer_review": buyer_review,
                    "seller_review": seller_review,
                    "buyer_apology": buyer_apology,
                    "seller_apology": seller_apology,
                }
                st.session_state.form_step = "survey"
                st.success("평가가 저장되었습니다. 다음 설문을 작성해주세요.")
                st.rerun()

    # 2단계: 심리적 만족도 설문 폼
    elif step == "survey":
        with st.form("survey_form"):
            st.subheader("심리적 만족도 설문")

            satisfaction = st.slider(
                "전반적인 협상 결과에 얼마나 만족하셨나요?",
                min_value=1,
                max_value=7,
                value=4,
            )

            fairness = st.slider(
                "이번 협상이 공정했다고 느끼셨나요?",
                min_value=1,
                max_value=7,
                value=4,
            )

            trust = st.slider(
                "상대방에 대한 신뢰 수준은 어떠신가요?",
                min_value=1,
                max_value=7,
                value=4,
            )

            willingness = st.slider(
                "비슷한 상황에서 다시 상대방과 협상하고 싶으신가요?",
                min_value=1,
                max_value=7,
                value=4,
            )

            comment = st.text_area(
                "추가 의견 (선택)",
                help="협상 경험 전반에 대한 피드백이나 느낀 점을 자유롭게 남겨주세요."
            )

            submitted = st.form_submit_button("설문 제출 및 저장")
            if submitted:
                st.session_state.survey_results = {
                    "satisfaction": satisfaction,
                    "fairness": fairness,
                    "trust": trust,
                    "willingness": willingness,
                    "comment": comment,
                }

                # 설문까지 완료되면 logging_node 실행 및 저장
                with st.spinner("협상 결과와 설문을 저장 중입니다..."):
                    try:
                        snapshot = st.session_state.graph.get_state(st.session_state.config)
                        current_state = snapshot.values
                        current_state["human_evaluation"] = st.session_state.human_evaluation
                        current_state["survey_results"] = st.session_state.survey_results

                        log_result = logging_node(current_state)
                        st.session_state.graph.update_state(
                            st.session_state.config,
                            log_result,
                        )

                        st.session_state.show_end_success = True
                        st.session_state.form_step = "done"
                        st.session_state.messages.append(
                            {
                                "role": "system",
                                "content": "협상이 종료되고 평가 및 설문이 저장되었습니다.",
                                "avatar": "✅",
                            }
                        )
                    except Exception as e:
                        st.error(f"❌ 저장 중 오류 발생: {e}")

                st.rerun()

    elif step == "done":
        st.success("✅ 평가와 설문이 모두 완료되었습니다.")

def render_chat_screen():
    """채팅 화면 전체를 구성하는 메인 함수"""
    
    # 1. 이전 협상 종료 성공 메시지 표시
    if st.session_state.get("show_end_success"):
        st.success("✅ 협상이 종료되고 평가 결과가 저장되었습니다!")
        st.balloons()
        st.session_state.show_end_success = False
    
    # 2. 사이드바 렌더링
    render_sidebar()

    # 3. 대화 기록 렌더링
    render_chat_history()

    # 4. 협상 종료 후라면 입력 대신 평가/설문 폼 렌더링
    step = st.session_state.get("form_step")

    if step in (None, "", "none"):
        # 아직 협상이 진행 중인 경우에만 채팅 입력 허용
        if prompt := st.chat_input("메시지를 입력하세요..."):
            # (1) 사용자 메시지 즉시 표시
            st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👤"})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # (2) AI 응답 처리 (스트리밍)
            with st.spinner("상대방이 생각 중입니다..."):
                # 여기서 복잡한 로직 함수 호출
                should_reset = process_graph_stream(prompt)
                
                if should_reset:
                    st.rerun()
            
            # (3) 종료 체크
            check_negotiation_finished()
    else:
        st.info("협상이 종료되었습니다. 아래 평가/설문 폼을 완료해주세요.")
        render_post_negotiation_forms()