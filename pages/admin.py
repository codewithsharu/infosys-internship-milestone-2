import streamlit as st
from backend.db_connection import get_all_users_with_roles, get_summarizer_history, get_paraphrase_history

def admin_page():
    st.title("👑 Admin Panel")

    st.subheader("Registered Users and Roles")
    users = get_all_users_with_roles()
    if users:
        users_data = []
        for user in users:
            users_data.append({"ID": user['id'], "Username": user['username'], "Role": user['role']})
        st.dataframe(users_data, use_container_width=True)
    else:
        st.info("No users registered yet.")

    st.subheader("User Activity History")
    if users:
        for user in users:
            st.markdown(f"#### History for User: {user['username']} (ID: {user['id']})")
            
            st.markdown("##### Summarization History")
            summarizer_history = get_summarizer_history(user['id'])
            if summarizer_history:
                for entry in summarizer_history:
                    feedback_icon = "👍" if entry['feedback_like'] else ("👎" if entry['feedback_like'] is False else "")
                    with st.expander(f"Summary (ID: {entry['id']}) from {entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S")} {feedback_icon}"):
                        st.write(f"**Original Text:** {entry['original_text']}")
                        st.write(f"**Summarized Text:** {entry['summarized_text']}")
                        if entry['feedback_text']:
                            st.write(f"**Feedback:** {entry['feedback_text']}")
            else:
                st.info("No summarization history for this user.")
            
            st.markdown("##### Paraphrase History")
            paraphrase_history = get_paraphrase_history(user['id'])
            if paraphrase_history:
                for entry in paraphrase_history:
                    feedback_icon = "👍" if entry['feedback_like'] else ("👎" if entry['feedback_like'] is False else "")
                    with st.expander(f"Paraphrase (ID: {entry['id']}) from {entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S")} {feedback_icon}"):
                        st.write(f"**Original Text:** {entry['original_text']}")
                        st.write(f"**Paraphrased Text:** {entry['paraphrased_text']}")
                        if entry['feedback_text']:
                            st.write(f"**Feedback:** {entry['feedback_text']}")
            else:
                st.info("No paraphrasing history for this user.")
            
            st.markdown("---") # Separator for better readability between users
    else:
        st.info("No user activity to display yet.")

    if st.button("Back to Summarizer"):
        st.session_state.page = "summarizer"
        st.rerun()
