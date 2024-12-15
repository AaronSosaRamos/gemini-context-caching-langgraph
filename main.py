from app.agent.agent import graph
import logging

if __name__ == "__main__":

    # Reusable function for graph invocation, logging, and writing results
    def process_step(step_number, description, initial_state, graph, config, results_file):
        try:
            # Log the initial state for traceability
            logger.info(f"Step {step_number} - {description} (INITIAL STATE): {initial_state}")
            
            # Invoke the graph
            result = graph.invoke(initial_state, config=config)
            
            # Extract messages and the last message content
            messages = result['messages']
            last_message_content = messages[-1].content

            # Log the complete result object for debugging purposes
            logger.info(f"Step {step_number} - {description} (RESULT): {result}")

            # Log step details
            print(f"Step {step_number} - {description} (MESSAGES): {messages}")
            logger.info(f"Step {step_number} - {description} (MESSAGES): {messages}")
            print(f"Step {step_number} - {description}: {last_message_content}")
            logger.info(f"Step {step_number} - {description}: {last_message_content}")

            # Write to results file
            results_file.write(f"Step {step_number} - {description} (MESSAGES):\n")
            results_file.write(str(messages) + '\n\n')
            results_file.write(f"Step {step_number} - {description}:\n")
            results_file.write(last_message_content + '\n\n')

            return result  # Return the result for further processing if needed
        except Exception as e:
            # Log the error with the step and state information
            logger.error(f"Error in Step {step_number} - {description} (STATE: {initial_state}): {str(e)}")
            raise e

    # Configure logging
    logging.basicConfig(
        filename='process_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Logger instance
    logger = logging.getLogger()

    # Reading the content of the file
    try:
        with open('genesis.txt', 'r') as file:
            content = file.read()
        logger.info("Successfully read 'genesis.txt'.")
    except FileNotFoundError:
        logger.error("File 'genesis.txt' not found.")
        content = None

    # Process steps
    try:
        with open('results.txt', 'w', encoding='utf-8') as results_file:
            # Step 0: Initial user input
            result = process_step(
                0, "Search in the Internet",
                {
                    "messages": [
                        {"role": "user", "content": "What is the weather in San Jerónimo de Tunán, Perú? Search in the Internet for it."}
                    ]
                },
                graph, {"model_name": "openai"}, results_file
            )

            # Step 1: Cache creation
            result = process_step(
                1, "Cache creation result",
                {
                    "messages": [
                        {"role": "user", "content": "Create a cache for the Genesis book."}
                    ],
                    "text": content 
                },
                graph, {"model_name": "openai"}, results_file
            )
            cache_name = result["cache_name"]

            # Step 2: Cache update
            result = process_step(
                2, "Cache update result",
                {
                    "messages": [
                        {"role": "user", "content": "Explicitly update the existing cache by setting its time-to-live (TTL) to 6 minutes."}
                    ],
                    "cache_name": cache_name,
                    "ttl": 6
                },
                graph, {"model_name": "openai"}, results_file
            )

            # Step 3: Query using cache
            result = process_step(
                3, "Query result",
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Use the cache to get the exact chapter 25 from Genesis."
                        }
                    ],
                    "cache_name": cache_name,
                    "decision": "get_cache",
                    "init_decision": True,
                    "query": "Use the cache to get the exact chapter 25 from Genesis."
                },
                graph, {"model_name": "openai"}, results_file
            )

            # Step 4: Cache deletion
            result = process_step(
                4, "Cache deletion result",
                {
                    "messages": [
                        {"role": "user", "content": "Delete the cache to avoid extra costs."}
                    ],
                    "cache_name": cache_name,
                    "decision": "delete_cache",
                    "init_decision": True
                },
                graph, {"model_name": "openai"}, results_file
            )

        logger.info("All steps successfully processed and results written to 'results.txt'.")
    except Exception as e:
        logger.error(f"Error during processing steps: {str(e)}")
