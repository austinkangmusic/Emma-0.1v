## Tools available:

### response:
Final answer for user.
Ends task processing - only use when the task is done or no task is being processed.
Place your result in "text" argument.
**Example usage**:
~~~json
{
    "thoughts": [
        "The user has greeted me...",
        "I will...",
    ],
    "tool_name": "response",
    "tool_args": {
        "text": "Hi...",
    }
}
~~~