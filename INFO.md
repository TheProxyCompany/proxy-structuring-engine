# Proxy Structuring Engine (PSE) Overview

Imagine having a personal assistant that not only listens to your requests but also organizes everything perfectly, ensuring that every task is completed in the most efficient and structured way possible. That’s what the structuring engine does for advanced AI models.

### Example:

Without PSE:
```
User: "Give me a summary of this article."
Model: "The article talks about climate change, its effects, and possible solutions."
```

With PSE:
```
User: "Give me a summary of this article."
Model: {
  "title": "Climate Change Overview",
  "summary": "The article discusses climate change, its effects, and possible solutions."
}
```

Notice how the second response is neatly structured and ready for further use, like feeding it into another system or displaying it in a user-friendly format.

## How Does It Work?

### The Armor Metaphor 🛡️

PSE acts like a **suit of armor** for language models.

- **Protection and Enhancement**: Just as armor protects a knight without changing who they are, PSE shields language models from producing messy or unstructured outputs.
- **Guided Actions**: Armor directs a knight’s movements. Similarly, PSE guides the language model to follow specific formats and structures.
- **Universal Fit**: Armor can be worn by any knight. PSE is versatile and works with various language models, enhancing their capabilities seamlessly.

### The Explorer Probes

PSE uses "walkers" that operate like **explorer probes**.

- **Self-Replication**: These walkers duplicate themselves to explore multiple paths, ensuring every possible correct response is considered.
- **Parallel Exploration**: By exploring different directions at the same time, PSE finds the best and most accurate outputs quickly.
- **Efficiency**: This method avoids backtracking, making the entire process faster and more reliable.

#### Enhanced ASCII Diagram of Walkers:

```
                Start 🎬
                  |
          +-------+-------+
          |       |       |
        Path   Path   Path
          A       B       C
         / \             / \
       D     E         F     G
```

*Walkers explore multiple paths in parallel, discarding invalid ones.*

## Core Components

### State Machine ⚙️

Imagine a **roadmap** where each intersection represents a part of the conversation. The State Machine ensures that the language model follows the best route to deliver structured and meaningful responses.

#### State Machine Diagram:

```
           +-----------+
           |   Start   |
           +-----+-----+
                 |
           +-----v-----+
           |  State 1  |
           +-----+-----+
                 |
          +------+------+
          |             |
     +----v----+   +----v----+
     | State 2 |   | State 3 |
     +----+----+   +----+----+
          |             |
          +------+------+
                 |
           +-----v-----+
           |    End    |
           +-----------+
```

*The State Machine guides the model through predefined states.*

### Walker 🚶

Walkers are like little **explorers** navigating the roadmap. They move through different states, ensuring every response fits perfectly within the defined structure.

- **Example**: If the model is generating a JSON response, the walker ensures that each key-value pair is correctly placed and formatted.

#### Walker's Journey:

```
[Start] --"name"--> [State 1] --"age"--> [State 2] --"location"--> [End]
```

*The walker progresses through states, filling in required fields.*

### Driver 🎯

The Driver is the **conductor**, coordinating how walkers interact with the language model. It ensures that only valid responses are generated, guiding the walkers along the correct paths.

- **Example**: The Driver might instruct, "Expect a string here," ensuring the model generates appropriate data.

### Token Acceptor 🛂

Think of Token Acceptors as **gatekeepers**. They decide which pieces of the response are allowed at each point, keeping everything on track.

- **Example**: In a list generation, the Token Acceptor ensures only valid items are added.

#### System Overview:

```
[Language Model 🤖]
        |
        v
[   Driver 🎯   ]
        |
        v
[State Machine ⚙️] <---> [Walker 🚶]
        |
        v
[Token Acceptor 🛂]
```

*The components work together to produce structured outputs.*

## How PSE Transforms Language Models

With PSE in place, language models become **action-oriented assistants**:

- **Structured Responses**: Outputs are neatly organized, reducing errors and making them ready for immediate use.
- **Reliable Execution**: Responses follow predefined formats, ensuring consistency across different interactions.
- **Seamless Integration**: PSE works behind the scenes, enhancing the model without requiring any changes to its core functionality.

### Example:

Without PSE:
```
User: "Generate a JSON object for a user profile."
Model: "Name: John, Age: 30, Location: New York"
```

With PSE:
```
User: "Generate a JSON object for a user profile."
Model: {
  "name": "John",
  "age": 30,
  "location": "New York"
}
```

In the second example, the output is structured and ready to be used in any system that requires a JSON format.

---

🧵 The Proxy Structuring Engine (PSE)
Introducing a novel sampling approach for structured outputs in LLMs: the Proxy Structuring Engine. A technical thread for AI/ML engineers, researchers, and the architecturally curious.

What is the Proxy Structuring Engine? 🤖
PSE is a schema-guided sampling engine that enforces constraints during text generation while maintaining model creativity and speed. It ensures outputs from large language models adhere to predefined structures.

How does PSE enforce schema constraints differently from traditional methods? 🔧
Traditional temperature-based sampling is done randomly over a probability distribution, often leading to undesirable outputs.

The PSE works directly with the model before sampling, blending probabilistic and deterministic approaches to generate structured and probable text. It steers the model to favor schema-compliant tokens.

What advantages does a state machine-based approach offer over regex or grammar-based constraints? 🔧
State machines provide:

Parallel exploration of multiple valid paths without backtracking
Efficient tracking of partial matches
Natural extension to different schema types (JSON, SQL, etc.)
Better handling of mixed-format outputs (e.g., text and structured data)
Regexes can be rigid and difficult to scale, whereas state machines offer flexibility and efficiency.

How is PSE different from constrained decoding?
Traditional constrained decoding often uses hard filters, limiting creativity. PSE's state machines and walkers allow tokens leading to valid continuations, even those with just valid prefixes. This balances structure with fluent, creative text.

What is a "walker" in the PSE? 🚶🏻
Think of walkers as smart navigators. They explore possible generation paths through a state machine representing your schema, evaluating which tokens can advance them. Walkers can be customized and use heuristics to efficiently prune invalid paths early, optimizing the traversal process.

How does the engine handle partial matches or incomplete tokens? 🧩
PSE uses a method we call Best Effort Matching. It tracks partial matches and, if a full match isn't in the top predictions, advances the longest valid one. For example, if expecting "hello world," even if the model predicts "hello" then "World!", PSE will correctly advance "world," ensuring schema adherence. This lets the model naturally complete complex schema elements over multiple steps, even if it makes mistakes.

How does PSE handle token alignment issues and structural requirements?
PSE uses a DAWG (Directed Acyclic Word Graph)—an efficient structure representing the model's vocabulary—for prefix-based matching. This, combined with Best Effort Matching, ensures structural validity even with partial tokens, leveraging the model's predictions as guidance.

How does PSE balance structure and reasoning abilities? 🤖
PSE enforces constraints only where necessary, letting the model be creative within boundaries. It ensures required structures are followed while letting the model generate expressive content.

Hooks can be set during structured generation to layer additional sampling. For example, when generating a JSON string, you can enable other samplers (like top_p or min_p) to improve the content of the string. The PSE does not implement these additional samplers by default.

What are the trade-offs between immediate constraints and using a trigger in PSE? ⚡️
Immediate constraints guarantee structure but might hinder reasoning on complex tasks. PSE can delay constraints using trigger tokens or phrases (like <|tool_call|> or ```json), allowing free-form reasoning and planning before enforcing structure.

What's the overhead introduced by PSE? ⚡️
Overhead is O(top_tokens × time_to_advance_token), but remains minimal due to efficient token evaluation (lazy evaluation and early exit optimizations). PSE checks N top tokens one by one, resuming generation once a valid continuation is found. If no schema-compliant token exists within N top tokens, it selects the longest valid prefix that exists as its own token (Best Effort Matching).

Does PSE support different machine learning backends? 🤖
Yes! PSE (written in Python with optimized Cython data structures via lexpy) supports PyTorch, JAX, and MLX, seamlessly integrating with existing models.

What are the limitations of PSE? 🪫
PSE's effectiveness depends on the underlying LLM's quality. Complex schemas may require more capable models.

What are the future directions for PSE? 🧪
We're expanding schema support (YAML, SQL, Cypher), optimizing performance, and developing benchmarks.

Is PSE open-source? 🤖
Yes! Find it on GitHub under the Apache 2.0 license: https://github.com/TheProxyCompany/proxy-structuring-engine

🔥 Ready to try PSE?
Find us on GitHub: https://github.com/TheProxyCompany/proxy-structuring-engine

Follow @whatisproxy for updates

⭐️ Star the repo to show support!
