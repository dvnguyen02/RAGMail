import streamlit as st
import sys
import os
import time
from datetime import datetime
import json
from pathlib import Path

# Import the SimpleRAGMail class from app.py
from app import SimpleRAGMail

# Configure Streamlit page
st.set_page_config(
    page_title="RAGMail",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/RAGMail',
        'Report a bug': 'https://github.com/yourusername/RAGMail/issues',
        'About': 'RAGMail - RAG for Emails'
    }
)

@st.cache_resource
def get_ragmail_instance():
    """Create an instance of SimpleRAGMail."""
    return SimpleRAGMail()

def display_email_list(emails, title="Emails", allow_selection=False):
    """Display a list of emails
    
    Args:
        emails: List of email dictionaries
        title: Title for the email section
        allow_selection: Whether to allow selecting emails for deletion
    
    Returns:
        List of selected email IDs if allow_selection is True, otherwise None
    """
    if not emails:
        st.info("No emails found.")
        return [] if allow_selection else None
    
    st.subheader(f"{title} ({len(emails)})")
    
    selected_emails = []
    
    for i, email in enumerate(emails):
        email_id = email.get("id", f"unknown-{i}")
        subject = email.get("Subject", "No subject")
        sender = email.get("From", "Unknown sender")
        
        # Add checkbox for selection if allowed
        if allow_selection:
            col1, col2 = st.columns([1, 11])
            with col1:
                is_selected = st.checkbox("", key=f"select_{email_id}", value=st.session_state.get(f"select_{email_id}", False))
                if is_selected:
                    selected_emails.append(email_id)
            
            with col2:
                with st.expander(f"{subject} - {sender}"):
                    st.markdown(f"**From:** {sender}")
                    st.markdown(f"**Date:** {email.get('Date', 'Unknown date')}")
                    
                    # Show relevance score if available
                    if "similarity_score" in email:
                        st.progress(float(email.get("similarity_score", 0)))
                        st.caption(f"Relevance: {email['similarity_score']:.2f}")
                    
                    # Show body with a max height to prevent long emails from taking too much space
                    st.markdown("**Content:**")
                    st.text_area("Email Content", value=email.get("Body", "No content"), height=200, key=f"email_{i}", label_visibility="collapsed")
        else:
            with st.expander(f"{subject} - {sender}"):
                st.markdown(f"**From:** {sender}")
                st.markdown(f"**Date:** {email.get('Date', 'Unknown date')}")
                
                # Show relevance score if available
                if "similarity_score" in email:
                    st.progress(float(email.get("similarity_score", 0)))
                    st.caption(f"Relevance: {email['similarity_score']:.2f}")
                
                # Show body with a max height to prevent long emails from taking too much space
                st.markdown("**Content:**")
                st.text_area("Email Content", value=email.get("Body", "No content"), height=200, key=f"email_{i}", label_visibility="collapsed")
    
    return selected_emails if allow_selection else None

def display_summary_results(summary):
    """Display email summary results in a streamlit-friendly format."""
    if not summary:
        st.info("No summary available.")
        return
    
    # Main summary
    st.subheader("Summary")
    summary_text = summary.get('summary', 'No summary available.')
    st.info(summary_text)
    
    # Display in columns
    col1, col2 = st.columns(2)
    
    # Categories
    with col1:
        categories = summary.get('categories', [])
        if categories:
            st.subheader("Categories")
            for category in categories:
                name = category.get('name', 'Unknown')
                count = category.get('count', 0)
                st.code(f"{name}: {count}")
    
    # Important emails
    with col2:
        important = summary.get('important', [])
        if important:
            st.subheader("Important Emails")
            for item in important:
                if isinstance(item, dict):
                    email_id = item.get('id', 'Unknown')
                    reason = item.get('reason', 'No reason provided')
                    st.warning(f"{email_id}: {reason}")
                else:
                    st.warning(item)

def display_llm_results(results):
    """Display LLM search results in a streamlit-friendly format."""
    if not results:
        st.info("No results available.")
        return
    
    llm_response = results.get("response", "")
    emails = results.get("emails", [])
    
    # Display LLM response
    st.subheader("AI Response")
    st.success(llm_response)
    
    # Display referenced emails
    if emails:
        display_email_list(emails, "Referenced Emails")

def delete_emails(ragmail, email_ids=None):
    """Delete emails from storage.
    
    Args:
        ragmail: RAGMail instance
        email_ids: List of email IDs to delete. If None, all emails will be deleted.
        
    Returns:
        int: Number of emails deleted
    """
    if email_ids is None:
        # Delete all emails
        count_before = ragmail.document_store.count()
        
        # Clear document store
        ragmail.document_store.clear()
        
        # Clear vector store
        ragmail.vector_store.clear()
        
        return count_before
    else:
        # Delete specific emails
        count_deleted = 0
        for email_id in email_ids:
            # Delete from document store
            if ragmail.document_store.delete(email_id):
                count_deleted += 1
                
            # Delete from vector store
            ragmail.vector_store.delete(email_id)
        
        return count_deleted

def main():
    # Initialize the RAGMail app
    with st.spinner("Initializing RAGMail..."):
        ragmail = get_ragmail_instance()
    
    # Check if initialization was successful
    if not ragmail.is_initialized:
        st.error("RAGMail failed to initialize. Check the logs for details.")
        return
    
    # App title
    st.title("📧 RAGMail")
    
    # Sidebar navigation
    st.sidebar.markdown("## 📋 Navigation")
    page = st.sidebar.radio(
        "Select Feature",
        ["Dashboard", "Email Search & QA", "Manage Emails"]
    )
    
    # Check email count
    email_count = ragmail.document_store.count()
    st.sidebar.markdown(f"### 📊 Statistics")
    st.sidebar.markdown(f"**Emails in database:** {email_count}")
    
    # Show the selected page
    if page == "Dashboard":
        st.markdown("## 📊 Dashboard")
        st.markdown("Welcome to RAGMail! Here you can sync emails and generate summaries.")
        
        # Display email stats
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📧 {email_count} emails in database")
            
            # Recent emails info
            recent_emails = ragmail.get_recent_emails(since_days=1)
            st.info(f"📬 {len(recent_emails)} new emails in the last 24 hours")
        
        with col2:
            if st.button("💫 Quick Sync (10 emails)"):
                with st.spinner("Syncing recent emails..."):
                    synced = ragmail.sync_recent_emails(limit=10)
                    st.success(f"Synced {synced} emails")
                    st.rerun()
        
        # Sync Emails Section
        st.markdown("## 🔄 Sync Emails")
        st.markdown("Fetch recent emails from your account and store them locally.")
        
        # Sync control
        limit = st.slider("Number of emails to fetch", min_value=5, max_value=100, value=30, step=5)
        
        if st.button("🔄 Sync Emails"):
            with st.spinner(f"Syncing {limit} emails..."):
                synced_count = ragmail.sync_recent_emails(limit=limit)
                
                if synced_count > 0:
                    st.success(f"Successfully synced {synced_count} emails!")
                else:
                    st.error("Failed to sync emails. Check the logs for details.")
        
        # Daily Summary Section
        st.markdown("## 📝 Daily Email Summary")
        st.markdown("Get a summary of your emails from yesterday.")
        
        if st.button("📝 Generate Daily Summary"):
            with st.spinner("Fetching emails from yesterday..."):
                recent_emails = ragmail.get_recent_emails(since_days=1)
                
                if recent_emails:
                    st.info(f"Found {len(recent_emails)} emails from yesterday")
                    
                    with st.spinner("Generating summary..."):
                        summary = ragmail.generate_daily_summary(recent_emails)
                        display_summary_results(summary)
                else:
                    st.warning("No emails found from yesterday")
    
    elif page == "Email Search & QA":
        st.markdown("## 🔍 Email Search & QA")
        st.markdown("Search your emails or ask questions about them.")
        
        # Explanation box for search types
        with st.expander("ℹ️ What's the difference between search types?"):
            st.markdown("""
            **Semantic Search** uses AI to understand the meaning behind your query and finds emails with similar meaning, 
            even if they don't contain the exact keywords. This is great for finding conceptually related emails.
            
            **Keyword Search** looks for exact word matches in emails. This is more like traditional search that finds 
            emails containing the specific words you typed.
            """)
        
        # Tabs for Search and Q&A
        search_tab, qa_tab = st.tabs(["Email Search", "Ask About Emails"])
        
        with search_tab:
            # Search query
            query = st.text_input("Search query", placeholder="Enter your search terms...")
            search_limit = st.slider("Maximum results", min_value=3, max_value=20, value=5)
            
            col1, col2 = st.columns(2)
            with col1:
                semantic_search = st.button("🔍 Semantic Search")
            with col2:
                keyword_search = st.button("🔤 Keyword Search")
            
            if semantic_search and query:
                with st.spinner(f"Searching for: {query}"):
                    results = ragmail.search_emails(query, top_k=search_limit)
                    
                    if results:
                        display_email_list(results, "Search Results")
                    else:
                        st.warning("No emails found matching your query.")
            
            elif keyword_search and query:
                with st.spinner(f"Keyword search for: {query}"):
                    results = ragmail.document_store.search(query)
                    results = results[:search_limit] if len(results) > search_limit else results
                    
                    if results:
                        display_email_list(results, "Keyword Search Results")
                    else:
                        st.warning("No emails found matching your query.")
                        
        with qa_tab:
            # Input for the question
            query = st.text_input("Question", placeholder="Ask a question about your emails...")
            
            if st.button("🤖 Ask") and query:
                with st.spinner("Searching emails and generating response..."):
                    results = ragmail.llm_search(query)
                    display_llm_results(results)
    
    elif page == "Manage Emails":
        st.markdown("## 🗑️ Manage Emails")
        st.markdown("Delete emails from the database.")
        
        # Section for deleting all emails
        st.warning("Delete All Emails")
        st.markdown("This will remove all emails from both the document store and vector store. This action cannot be undone.")
        
        # Add confirmation for delete all
        delete_all = st.button("🗑️ Delete All Emails")
        
        if delete_all:
            delete_confirm = st.checkbox("I understand this action cannot be undone")
            
            if delete_confirm:
                if st.button("Confirm Delete All"):
                    with st.spinner("Deleting all emails..."):
                        deleted_count = delete_emails(ragmail)
                        st.success(f"Successfully deleted {deleted_count} emails!")
                        st.rerun()
        
        st.markdown("---")
        
        # Section for selectively deleting emails
        st.markdown("### Select Emails to Delete")
        st.markdown("Use the checkboxes to select which emails you want to delete.")
        
        # Select time range for listing emails
        days_range = st.slider("Show emails from the past days", min_value=1, max_value=30, value=7)
        
        with st.spinner(f"Loading emails from the past {days_range} days..."):
            emails_to_show = ragmail.get_recent_emails(since_days=days_range)
            
            if not emails_to_show:
                st.info(f"No emails found from the past {days_range} days.")
            else:
                st.info(f"Found {len(emails_to_show)} emails. Select the ones you want to delete.")
                
                # Add Select All button
                if st.button("Select All Emails"):
                    st.session_state["select_all"] = not st.session_state.get("select_all", False)
                    for i, email in enumerate(emails_to_show):
                        email_id = email.get("id", f"unknown-{i}")
                        st.session_state[f"select_{email_id}"] = st.session_state.get("select_all", False)
                    st.rerun()
                
                # Display emails with selection checkboxes
                selected_ids = display_email_list(emails_to_show, "Emails to Manage", allow_selection=True)
                
                if selected_ids:
                    if st.button(f"Delete {len(selected_ids)} Selected Emails"):
                        with st.spinner(f"Deleting {len(selected_ids)} emails..."):
                            deleted_count = delete_emails(ragmail, selected_ids)
                            st.success(f"Successfully deleted {deleted_count} emails!")
                            st.rerun()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("RAGMail - RAG for Emails")
    st.sidebar.caption(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
