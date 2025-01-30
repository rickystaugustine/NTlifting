import requests
import json
import csv
import pandas as pd
import os

# ✅ Define Repository Information
GITHUB_TOKEN = os.getenv("GH_PAT")
OWNER = "rickystaugustine"
REPO = "NTlifting"
PROJECT_NUMBER = 1
GRAPHQL_URL = "https://api.github.com/graphql"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

# ✅ Optimized GraphQL Query to Fetch Issues & Kanban Board
query = """
{
  user(login: "rickystaugustine") {
    projectV2(number: 1) {
      title
      url
      items(first: 50) {
        nodes {
          id
          fieldValues(first: 10) {
            nodes {
              ... on ProjectV2ItemFieldTextValue {
                text
              }
              ... on ProjectV2ItemFieldSingleSelectValue {
                name
                field {
                  ... on ProjectV2FieldCommon {
                    name
                  }
                }
              }
            }
          }
          content {
            ... on Issue {
              number
              title
              url
              state
              labels(first: 5) {
                nodes {
                  name
                }
              }
            }
          }
        }
      }
      fields(first: 10) {
        nodes {
          ... on ProjectV2Field {
            name
          }
        }
      }
    }
  }
}
"""

# ✅ Send GraphQL Request
response = requests.post(GRAPHQL_URL, json={"query": query}, headers=HEADERS)
data = response.json()

# 🔍 Debug: Print Full Response
print("\n🔍 **Full Response from GitHub:**")
print(json.dumps(data, indent=2))

# ✅ Check for Authentication Errors
if "message" in data and "status" in data and data["status"] == 401:
    print("\n🚨 **Error: Authentication failed (401 Unauthorized). Check your GH_PAT token permissions.**")
    exit(1)

# ✅ Extract Project Data
if "data" in data and "user" in data["data"] and "projectV2" in data["data"]["user"]:
    project = data["data"]["user"]["projectV2"]
    print(f"\n📌 **Project Title:** {project['title']}")
    print(f"🔗 **Project URL:** {project['url']}\n")

    # ✅ Extract Issue Data
    issues_list = []
    kanban_board_data = []

    for item in project["items"]["nodes"]:
        if "content" in item and item["content"] is not None:
            # ✅ Ensure 'number' exists before accessing it
            if "number" in item["content"]:
                issue_data = {
                    "Issue Number": item["content"]["number"],
                    "Title": item["content"]["title"],
                    "Status": item["content"]["state"],
                    "Labels": ", ".join([label["name"] for label in item["content"].get("labels", {}).get("nodes", [])]),
                    "URL": item["content"]["url"],
                }
                issues_list.append(issue_data)
                print(f"- #{issue_data['Issue Number']}: {issue_data['Title']} ({issue_data['Status']})")
                print(f"  🔗 {issue_data['URL']}\n")
            else:
                print(f"⚠️ Skipping non-issue item in project: {item['content']}")  # Debugging

        # ✅ Debug Kanban Board Data
        print("\n🛠️ Debugging: fieldValues for Item:")
        print(json.dumps(item.get("fieldValues", {}), indent=2))

        # ✅ Extract Kanban Board Data
        if "fieldValues" in item and "nodes" in item["fieldValues"]:
            kanban_column = None
            for field in item["fieldValues"]["nodes"]:
                if "name" in field and "field" in field and "name" in field["field"]:
                    if "Status" in field["field"]["name"]:  # Adjust based on actual API response
                        kanban_column = field["name"]

            if kanban_column and "content" in item and item["content"] is not None:
                kanban_board_data.append({
                    "Column": kanban_column,  # Kanban board column name
                    "Issue": item["content"].get("title", "Unknown")
                })
                print(f"✅ Assigned Issue '{item['content']['title']}' to Column: {kanban_column}")

    # ✅ Convert Issues Data to DataFrame
    df_issues = pd.DataFrame(issues_list)
    df_issues.to_csv("issues_export.csv", index=False)
    df_issues.to_excel("issues_export.xlsx", index=False)
    print("✅ GitHub Issues exported successfully.")

    # ✅ Convert Kanban Board Data to DataFrame
    if kanban_board_data:
        df_kanban = pd.DataFrame(kanban_board_data)
        df_kanban.to_csv("kanban_board.csv", index=False)
        df_kanban.to_excel("kanban_board.xlsx", index=False)
        print("✅ Kanban Board data retrieved successfully.")
    else:
        print("⚠️ No Kanban board data found. Check the API response and column names.")

else:
    print("\n🚨 **Error: 'data' field not found in response.** Check API permissions and request syntax.")
