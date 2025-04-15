# Bug Investigation Memo (BIM)

## Issue Details
- **Ticket ID**: {{ticket_id}}
- **Title**: {{title}}
- **Status**: {{status}}
- **Priority**: {{priority}}
- **Created**: {{created_at}}
- **Last Updated**: {{updated_at}}

## Problem Description
{{description}}

## Environment
- **Service**: {{environment.service}}
- **Version**: {{environment.version}}
- **Cluster**: {{environment.cluster}}

## Investigation Summary
{{summary}}

## Logs Analysis
{{#logs}}
- **Error Rate**: {{error_rate}}
- **Response Time**: {{response_time}}
{{/logs}}

## Related Issues
{{#related_issues}}
- [{{issue_id}}] {{title}} (Similarity: {{similarity_score}})
  - Resolution: {{resolution}}
{{/related_issues}}

## Metrics Overview
{{#metrics}}
- CPU Usage: {{cpu_usage_summary}}
- Memory Usage: {{memory_usage_summary}}
- Error Rate: {{error_rate_summary}}
{{/metrics}}

## Traces Analysis
{{#traces}}
- {{trace_id}}: {{status}}
  {{#spans}}
  - {{service}}.{{operation}}: {{duration_ms}}ms ({{status}})
  {{/spans}}
{{/traces}}

## Next Steps
1. [ ] Review error patterns in logs
2. [ ] Analyze performance metrics
3. [ ] Check similar past incidents
4. [ ] Identify potential fixes 