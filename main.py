from app.agent.agent import graph
import logging

if __name__ == "__main__":

    # Reusable function for graph invocation, logging, and writing results
    def process_step(step_number, description, initial_state, graph, config, results_file):
        try:
            # Invoke the graph
            result = graph.invoke(initial_state, config=config)
            
            # Extract messages and the last message content
            messages = result['messages']
            last_message_content = messages[-1].content

            # Logging step details
            logger.info(f"Step {step_number} - {description} (MESSAGES): {messages}")
            logger.info(f"Step {step_number} - {description}: {last_message_content}")

            # Write to results file
            results_file.write(f"Step {step_number} - {description} (MESSAGES):\n")
            results_file.write(str(messages) + '\n\n')
            results_file.write(f"Step {step_number} - {description}:\n")
            results_file.write(last_message_content + '\n\n')

            return result  # Return the result for further processing if needed
        except Exception as e:
            logger.error(f"Error in Step {step_number} - {description}: {str(e)}")
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

    # Steps to process
    steps = [
        (0, "Tavily imp.", {
            "messages": [
                {"role": "user", "content": "What is the weather in San Jerónimo de Tunán, Perú?"}
            ]
        }),
        (1, "Cache creation result", {
            "messages": [
                {"role": "user", "content": f"Create a cache for the Genesis book: {content}."}
            ]
        }),
        (2, "Cache update result", {
            "messages": [
                {"role": "user", "content": "Update the cache time-to-live to 2 minutes."}
            ]
        }),
        (3, "Query result", {
            "messages": [
                {"role": "user", "content": "How do I reset my device? Use the cached context."}
            ]
        }),
        (4, "Cache deletion result", {
            "messages": [
                {"role": "user", "content": "Delete the cache to avoid extra costs."}
            ]
        })
    ]

    # Process each step
    try:
        with open('results.txt', 'w', encoding='utf-8') as results_file:
            for step_number, description, initial_state in steps:
                process_step(step_number, description, initial_state, graph, {"model_name": "openai"}, results_file)
        logger.info("All steps successfully processed and results written to 'results.txt'.")
    except Exception as e:
        logger.error(f"Error during processing steps: {str(e)}")