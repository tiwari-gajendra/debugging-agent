# Amazon Bedrock Integration Guide

This guide explains how to set up and use Amazon Bedrock with the Debugging Agents system for LLM-powered debugging.

## Prerequisites

Before you begin, ensure you have:

1. An AWS account with access to Amazon Bedrock
2. AWS CLI installed and configured 
3. Appropriate permissions to use Amazon Bedrock models (particularly Anthropic Claude models)
4. The `langchain-aws` package installed (`pip install -U langchain-aws`)

## Setup

### 1. Configure AWS Credentials

First, make sure your AWS credentials are properly configured. You can do this via environment variables or AWS configuration files:

```bash
# Option 1: Set environment variables (temporary)
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1  # or your preferred region

# Option 2: Configure AWS CLI (persistent)
aws configure
```

When using the AWS CLI configuration, you'll be prompted to enter your AWS Access Key ID, AWS Secret Access Key, default region, and output format.

### 2. Verify Bedrock Access

Ensure you have been granted access to Amazon Bedrock and the models you intend to use. You can verify access by listing models:

```bash
aws bedrock list-foundation-models
```

If this command succeeds, you have the necessary permissions to use Bedrock.

### 3. Update Environment Variables

Edit your `.env` file in the debugging-agents project root to use Bedrock:

```bash
# LLM Provider Configuration
LLM_PROVIDER=bedrock
BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0

# AWS Configuration (if not set globally)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

Available models include:
- `anthropic.claude-3-sonnet-20240229-v1:0` (recommended)
- `anthropic.claude-3-haiku-20240307-v1:0` (faster but less capable)
- `anthropic.claude-instant-v1`
- `meta.llama3-70b-instruct-v1:0`
- Check AWS console for the latest models in your region

## Usage

Once configured, you can use the Debugging Agents system with Bedrock by simply specifying it as the provider:

```bash
# Run with Bedrock as the provider
debug-agent debug YOUR-ISSUE-123 --llm-provider bedrock

# Or use the provider from your .env file (if set to bedrock)
debug-agent debug YOUR-ISSUE-123
```

## Testing Your Bedrock Setup

You can verify that your Bedrock configuration is working by running the info command:

```bash
debug-agent info
```

This should display your current configuration, including the LLM provider (bedrock) and the model you've configured.

To perform a simple test of the Bedrock integration:

```bash
debug-agent debug TEST-BEDROCK-CONFIG --llm-provider bedrock
```

If the command executes without authentication errors, your Bedrock configuration is working correctly.

## Implementation Details

The Debugging Agents system uses the `ChatBedrock` class from the `langchain_aws` package to interact with Amazon Bedrock. The integration is configured in the `LLMFactory` class:

```python
# In src/utils/llm_factory.py
from langchain_aws import ChatBedrock

# ...

# For Bedrock provider
model = model_name or os.getenv('BEDROCK_MODEL', 'anthropic.claude-3-sonnet-20240229-v1:0')
region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

# Use proper model format for Bedrock
bedrock_model = model
if not model.startswith('bedrock/'):
    bedrock_model = f"bedrock/{model}"
    
# Handle temperature correctly in model_kwargs
bedrock_kwargs = kwargs.copy()
if 'temperature' in bedrock_kwargs:
    del bedrock_kwargs['temperature']
    
model_kwargs = bedrock_kwargs.pop('model_kwargs', {})
model_kwargs['temperature'] = temperature
    
return ChatBedrock(
    model_id=model,
    region_name=region,
    model_kwargs=model_kwargs,
    **bedrock_kwargs
)
```

Note: Bedrock models require the temperature parameter to be specified in `model_kwargs` rather than as a direct parameter due to validation requirements in the latest LangChain implementation.

## Recent Fixes

### Temperature Parameter Validation Fix

The Debugging Agents system recently addressed a validation error with the temperature parameter in the Bedrock integration:

```
Error: 1 validation error for BedrockChat
temperature
  Extra inputs are not permitted [type=extra_forbidden, input_value=0.2, input_type=float]
```

This error occurs because the newer version of `langchain_aws.ChatBedrock` doesn't accept temperature as a direct parameter. The fix involves:

1. Removing the temperature from the direct arguments to ChatBedrock
2. Adding the temperature to the model_kwargs dictionary instead
3. Ensuring proper format for Bedrock model identifiers

### CrewAI Integration Fix

For proper integration with CrewAI when using Bedrock:

```python
# In crew_manager.py
if self.provider == 'bedrock':
    # This helps CrewAI know we're using Bedrock
    os.environ["CREW_LLM_PROVIDER"] = "bedrock"
```

This ensures that CrewAI correctly recognizes and works with the Bedrock LLM provider.

## Troubleshooting

### Missing Credentials

If you receive an error about missing credentials:

```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

Make sure you've properly set up your AWS credentials either through environment variables or the AWS CLI.

### Access Denied

If you see an access denied error:

```
botocore.exceptions.ClientError: An error occurred (AccessDeniedException) when calling the InvokeModel operation: User is not authorized to perform bedrock:InvokeModel
```

Ensure that:
1. Your AWS account has access to Bedrock
2. You've requested and received access to the specific model you're trying to use
3. Your IAM user/role has the necessary permissions (bedrock:InvokeModel)

### Region Issues

If you encounter region-related errors:

```
botocore.exceptions.EndpointConnectionError: Could not connect to the endpoint URL
```

Verify that:
1. You've specified a valid AWS region where Bedrock is available (e.g., us-east-1, us-west-2)
2. The Bedrock model you're trying to use is available in that region

### Model Not Found

If the model you specified isn't available:

```
botocore.exceptions.ClientError: An error occurred (ModelNotFound) when calling the InvokeModel operation
```

Check that:
1. You've specified the correct model ID
2. The model is available in your region
3. You have access to the specific model you're trying to use

### Package Issues

If you see errors related to missing modules or validation errors:

```
ImportError: cannot import name 'ChatBedrock' from 'langchain_aws'
```

Make sure you've installed the latest version of the required packages:

```bash
pip install -U langchain-aws boto3
```

### LiteLLM Errors

If you encounter errors with LiteLLM:

```
litellm.BadRequestError: LLM Provider NOT provided. Pass in the LLM provider you are trying to call.
```

These errors can occur when using older versions of langchain and litellm. Update both packages:

```bash
pip install -U litellm langchain-aws crewai
```

## Advanced Configuration

For advanced users, you can customize the Bedrock configuration by modifying the `LLMFactory` class in `src/utils/llm_factory.py`. This might include:

- Setting custom model parameters
- Adding retries or timeouts
- Implementing model fallbacks

Example of advanced configuration via environment variables:

```bash
# Advanced Bedrock configuration
BEDROCK_INFERENCE_TIMEOUT=30  # Timeout in seconds
BEDROCK_MAX_TOKENS=4096       # Maximum tokens in the response
TEMPERATURE=0.5               # Control randomness (0.0-1.0)
```

You can also pass additional model-specific parameters:

```python
# Example of passing model-specific parameters in your code
model_kwargs = {
    "temperature": 0.5,
    "max_tokens": 4096,
    "top_p": 0.9,
    "stop_sequences": ["Human:", "Assistant:"]
}

llm = LLMFactory.create_llm("bedrock", model_kwargs=model_kwargs)
``` 