import requests
import json
import csv
import pandas as pd
import os

GITHUB_TOKEN = os.getenv("GH_PAT")
GRAPHQL_URL = "https://api.github.com/graphql"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

# ✅ Query using "number: 1" (Project Number, not ID)
query = """
{
  user(login: "rickystaugustine") {
    projectV2(number: 1) {
      title
      url
      items(first: 50) {
        nodes {
          id
          content {
            ... on Issue {
              number
              title
              url
              state
            }
            ... on PullRequest {
              number
              title
              url
              state
            }
          }
        }
      }
    }
  }
}
"""

# Send Request
response = requests.post(GRAPHQL_URL, json={"query": query}, headers=HEADERS)
data = response.json()

# 🔍 Debug: Print Full Response
print("\n🔍 **Full Response from GitHub:**")
print(json.dumps(data, indent=2))

# ✅ Extract Project Data Safely
if "data" in data and "user" in data["data"] and "projectV2" in data["data"]["user"]:
    project = data["data"]["user"]["projectV2"]
    print(f"\n📌 **Project Title:** {project['title']}")
    print(f"🔗 **Project URL:** {project['url']}\n")

    # Store issues in a list
    issues_list = []

    for item in project["items"]["nodes"]:
        if "content" in item and item["content"] is not None:
            issue_data = {
                "Issue Number": item["content"]["number"],
                "Title": item["content"]["title"],
                "Status": item["content"]["state"],
                "URL": item["content"]["url"],
            }
            issues_list.append(issue_data)
            print(f"- #{issue_data['Issue Number']}: {issue_data['Title']} ({issue_data['Status']})")
            print(f"  🔗 {issue_data['URL']}\n")
        else:
            print(f"- 📝 **Unlinked Task** (ID: {item['id']})")

    # Convert to DataFrame
    df = pd.DataFrame(issues_list)

    # Save to CSV
    csv_filename = "issues_export.csv"
    df.to_csv(csv_filename, index=False)
    print(f"✅ GitHub Issues exported to {csv_filename}")

    # Save to Excel
    excel_filename = "issues_export.xlsx"
    df.to_excel(excel_filename, index=False)
    print(f"✅ GitHub Issues exported to {excel_filename}")

else:
    print("\n🚨 **Error: 'data' field not found in response.** Check API permissions and request syntax.")
