# v0.20 interrupted by slow prompt prefill

The single-slot run reduced the first logical batch from 2,048 to 512 tokens. llama.cpp eventually completed that batch in 222.30 seconds (2.30 prompt tokens/second), but made no further reported progress through the next batch for over five continuous minutes. The preregistered stop rule fired before any Director response completed. No message was parsed or delivered, and no task outcome exists.

The server used one slot and 512/128 logical/physical batches; the model artifact and endpoint were verified. This shows the first resource reduction was insufficient on this host. It does not establish whether context-cache allocation, GPU sharing, or another backend factor caused the remaining delay. The incomplete response is not scored. A fresh diagnostic will reduce context allocation while keeping the same task/model and one-slot schedule.
