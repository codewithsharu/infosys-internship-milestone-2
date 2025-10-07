import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from db_connection import (
    get_connection,
    create_user_with_role,
    validate_user,
    get_user_id_and_role,
    insert_summarizer_history,
    update_summarizer_feedback,
    get_summarizer_history,
    insert_paraphrase_history,
    update_paraphrase_feedback,
    get_paraphrase_history,
    get_all_users_with_roles
)

def run_db_tests():
    print("--- Starting Database Debug Tests ---")

    # Test 1: Create a dummy user (if not exists)
    print("\n--- Test: User Creation and Login ---")
    test_username = "testuser_debug"
    test_password = "password123"
    dummy_user_id = None
    
    user_info = get_user_id_and_role(test_username)
    if user_info:
        print(f"User '{test_username}' already exists with ID: {user_info['id']} and role: {user_info['role']}")
        dummy_user_id = user_info['id']
    else:
        print(f"Creating user '{test_username}'...")
        if create_user_with_role("Test Debug", "test@debug.com", test_username, test_password, "user"):
            print(f"User '{test_username}' created successfully.")
            user_info = get_user_id_and_role(test_username)
            if user_info:
                dummy_user_id = user_info['id']
                print(f"Retrieved new user ID: {dummy_user_id}")
            else:
                print("Failed to retrieve user ID after creation.")
        else:
            print("Failed to create user.")

    # Test 2: Summarizer History Insertion (Logged-in user)
    print("\n--- Test: Summarizer History (Logged-in) ---")
    if dummy_user_id:
        original_text_sum = "This is a long text for summarization testing."
        summarized_text_sum = "This is a summary."
        print(f"Inserting summarizer history for user ID: {dummy_user_id}")
        sum_history_id = insert_summarizer_history(dummy_user_id, original_text_sum, summarized_text_sum)
        print(f"Summarizer history ID: {sum_history_id}")
        if sum_history_id:
            print("Updating summarizer feedback (Like)...")
            update_sum_like_status = update_summarizer_feedback(sum_history_id, True)
            print(f"Update like status: {update_sum_like_status}")
            print("Updating summarizer feedback (Text)...")
            update_sum_text_status = update_summarizer_feedback(sum_history_id, None, "Good summary!")
            print(f"Update text status: {update_sum_text_status}")
        else:
            print("Failed to insert summarizer history.")
    else:
        print("Skipping logged-in summarizer history test (no dummy user ID).")

    # Test 3: Summarizer History Insertion (Logged-out user - user_id = None)
    print("\n--- Test: Summarizer History (Logged-out) ---")
    original_text_sum_anon = "Another text for anonymous summarization."
    summarized_text_sum_anon = "Anonymous summary."
    print(f"Inserting summarizer history for anonymous user (ID: None)")
    sum_history_id_anon = insert_summarizer_history(None, original_text_sum_anon, summarized_text_sum_anon)
    print(f"Anonymous summarizer history ID: {sum_history_id_anon}")
    if sum_history_id_anon:
        print("Updating anonymous summarizer feedback (Unlike)...")
        update_sum_anon_unlike_status = update_summarizer_feedback(sum_history_id_anon, False)
        print(f"Anonymous update unlike status: {update_sum_anon_unlike_status}")
    else:
        print("Failed to insert anonymous summarizer history.")

    # Test 4: Paraphrase History Insertion (Logged-in user)
    print("\n--- Test: Paraphrase History (Logged-in) ---")
    if dummy_user_id:
        original_text_para = "Original sentence for paraphrasing."
        paraphrased_text_para = "A rephrased sentence for testing."
        print(f"Inserting paraphrase history for user ID: {dummy_user_id}")
        para_history_id = insert_paraphrase_history(dummy_user_id, original_text_para, paraphrased_text_para)
        print(f"Paraphrase history ID: {para_history_id}")
        if para_history_id:
            print("Updating paraphrase feedback (Like)...")
            update_para_like_status = update_paraphrase_feedback(para_history_id, True)
            print(f"Update like status: {update_para_like_status}")
        else:
            print("Failed to insert paraphrase history.")
    else:
        print("Skipping logged-in paraphrase history test (no dummy user ID).")

    # Test 5: Paraphrase History Insertion (Logged-out user - user_id = None)
    print("\n--- Test: Paraphrase History (Logged-out) ---")
    original_text_para_anon = "Another sentence for anonymous paraphrasing."
    paraphrased_text_para_anon = "Anonymously rephrased text."
    print(f"Inserting paraphrase history for anonymous user (ID: None)")
    para_history_id_anon = insert_paraphrase_history(None, original_text_para_anon, paraphrased_text_para_anon)
    print(f"Anonymous paraphrase history ID: {para_history_id_anon}")
    if para_history_id_anon:
        print("Updating anonymous paraphrase feedback (Text)...")
        update_para_anon_text_status = update_paraphrase_feedback(para_history_id_anon, None, "Good paraphrase!")
        print(f"Anonymous update text status: {update_para_anon_text_status}")
    else:
        print("Failed to insert anonymous paraphrase history.")

    print("\n--- Database Debug Tests Finished ---")

if __name__ == "__main__":
    run_db_tests()
