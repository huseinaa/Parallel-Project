%%sh
cat > sequential.c << EOF

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

#define MAX_PARAGRAPHS 100
#define EMBEDDING_SIZE 512
#define RESPONSE_BUFFER_SIZE 20000

typedef struct {
    double data[EMBEDDING_SIZE];
} Embedding;

typedef struct {
    char paragraphs[MAX_PARAGRAPHS][1024];
    int count;
} Paragraphs;

size_t write_response(void *ptr, size_t size, size_t nmemb, char *stream) {
    size_t real_size = size * nmemb;
    size_t current_length = strlen(stream);
    if (current_length + real_size < RESPONSE_BUFFER_SIZE - 1) {
        memcpy(stream + current_length, ptr, real_size);
        stream[current_length + real_size] = '\0';
    }
    return real_size;
}

void split_into_paragraphs(const char *text, Paragraphs *result) {
    const char *start = text;
    int count = 0;
    while (*start && count < MAX_PARAGRAPHS) {
        const char *end = strchr(start, '\n');
        if (!end) end = start + strlen(start);

        int len = end - start;
        if (len > 0 && len < 1024) {
            strncpy(result->paragraphs[count], start, len);
            result->paragraphs[count][len] = '\0';
            count++;
        }
        if (*end == '\0') break;
        start = end + 1;
    }
    result->count = count;
}

Embedding get_embedding(const char *text, CURL *curl) {
    CURLcode res;
    char response[RESPONSE_BUFFER_SIZE] = {0};
    Embedding embedding = {{0}};

    char data[2048];
    snprintf(data, sizeof(data), "{\"input\": \"%s\", \"model\": \"text-embedding-3-small\"}", text);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Authorization: Bearer");

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_URL, "https://api.openai.com/v1/embeddings");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_response);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);

    res = curl_easy_perform(curl);
    if (res == CURLE_OK) {
        char *ptr = strstr(response, "\"embedding\": [");
        if (ptr) {
            ptr += 14; 
            for (int i = 0; i < EMBEDDING_SIZE && *ptr != ']' && i < EMBEDDING_SIZE; i++) {
                embedding.data[i] = strtod(ptr, &ptr);
                while (*ptr == ',' || isspace(*ptr)) ptr++;
            }
        }
    } else {
        fprintf(stderr, "CURL Error: %s\n", curl_easy_strerror(res));
    }

    curl_slist_free_all(headers);
    return embedding;
}

double calculate_cosine_similarity(const Embedding *vec1, const Embedding *vec2) {
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        dot += vec1->data[i] * vec2->data[i];
        denom_a += vec1->data[i] * vec1->data[i];
        denom_b += vec2->data[i] * vec2->data[i];
    }
    return denom_a == 0.0 || denom_b == 0.0 ? 0 : dot / (sqrt(denom_a) * sqrt(denom_b));
}

int main() {
    int num_procs, rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    CURL *curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "Curl initialization failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    time_t start_time = time(NULL); // Record the start time

  char text_to_check[] = "What is generative AI?\n\n"
  "Generative AI refers to deep-learning models that can generate high-quality text, images, and other content based on the data they were trained on.\n\n"
  "Artificial intelligence has gone through many cycles of hype, but even to skeptics, the release of ChatGPT seems to mark a turning point. OpenAI's chatbot, powered by its latest large language model, can write poems, tell jokes, and churn out essays that look like a human created them.\n"
  "Prompt ChatGPT with a few words, and out comes love poems in the form of Yelp reviews, or song lyrics in the style of Nick Cave.\n";

  char reference_text[] = "What is generative AI?\n\n"
  "Generative artificial intelligence (AI) describes algorithms (such as ChatGPT) that can be used to create new content, including audio, code, images, text, simulations, and videos.\n"
  "Recent breakthroughs in the field have the potential to drastically change the way we approach content creation.\n\n"
  "Generative AI systems fall under the broad category of machine learning, and here's how one such system—ChatGPT—describes what it can do:\n\n"
  "Ready to take your creativity to the next level? Look no further than generative AI!\n"
  "This nifty form of machine learning allows computers to generate all sorts of new and exciting content, from music and art to entire virtual worlds. And it's not just for fun—generative AI has plenty of practical uses too, like creating new product designs and optimizing business processes. So why wait? Unleash the power of generative AI and see what amazing creations you can come up with!\n\n"
  "Did anything in that paragraph seem off to you? Maybe not. The grammar is perfect, the tone works, and the narrative flows.\n";

    Paragraphs input_paragraphs, reference_paragraphs;
    split_into_paragraphs(text_to_check, &input_paragraphs);
    split_into_paragraphs(reference_text, &reference_paragraphs);

    double total_similarity = 0.0;
    int comparisons = 0;

    // Divide work among processes
    int items_per_proc = input_paragraphs.count / num_procs;
    int start = rank * items_per_proc;
    int end = (rank == num_procs - 1) ? input_paragraphs.count : start + items_per_proc;

    for (int i = start; i < end; i++) {
        Embedding emb1 = get_embedding(input_paragraphs.paragraphs[i], curl);
        for (int j = 0; j < reference_paragraphs.count; j++) {
            Embedding emb2 = get_embedding(reference_paragraphs.paragraphs[j], curl);
            double similarity = calculate_cosine_similarity(&emb1, &emb2);
            total_similarity += similarity;
            comparisons++;
        }
    }

    double global_similarity;
    int global_comparisons;

    // Gather results at the root
    MPI_Reduce(&total_similarity, &global_similarity, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comparisons, &global_comparisons, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (global_comparisons > 0) {
            double average_similarity = (global_similarity / global_comparisons) * 100;
            printf("Plagiarism Score: %.2f%%\n", average_similarity);

            time_t end_time = time(NULL); // Record the end time
            printf("Runtime: %ld seconds\n", end_time - start_time);
        } else {
            printf("No valid comparisons made.\n");
        }
    }

    curl_easy_cleanup(curl);
    MPI_Finalize();
    return 0;
}

EOF
ls -l
