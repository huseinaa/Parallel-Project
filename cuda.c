%%sh
cat > sequential.cu << EOF

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <cuda_runtime.h>

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
    headers = curl_slist_append(headers, "Authorization: Bearer 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");

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

__global__ void calculate_cosine_similarity_cuda(double *vec1, double *vec2, double *output, int num_input, int num_ref, int embedding_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_input * num_ref;

    if (idx < total_pairs) {
        int i = idx / num_ref;  // Index for input_paragraphs
        int j = idx % num_ref;  // Index for reference_paragraphs
        double dot = 0.0, denom_a = 0.0, denom_b = 0.0;

        for (int k = 0; k < embedding_size; k++) {
            double v1 = vec1[i * embedding_size + k];
            double v2 = vec2[j * embedding_size + k];
            dot += v1 * v2;
            denom_a += v1 * v1;
            denom_b += v2 * v2;
        }

        if (denom_a != 0.0 && denom_b != 0.0) {
            double sim = dot / (sqrt(denom_a) * sqrt(denom_b));
            output[idx] = sim;
        } else {
            output[idx] = 0.0;
        }
    }
}

void launch_cosine_similarity_kernel(double *d_vec1, double *d_vec2, double *d_output, int num_input, int num_ref, int embedding_size) {
    int num_pairs = num_input * num_ref;
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_pairs + threadsPerBlock - 1) / threadsPerBlock;
    calculate_cosine_similarity_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_vec1, d_vec2, d_output, num_input, num_ref, embedding_size);
    cudaDeviceSynchronize(); 
}

int main() {
    time_t start_time = time(NULL);
    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL *curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "Curl initialization failed\n");
        return 1;
    }

    // Define input text for both the paragraphs to be checked
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

    // Split text into paragraphs and calculate embeddings
    Paragraphs input_paragraphs, reference_paragraphs;
    split_into_paragraphs(text_to_check, &input_paragraphs);
    split_into_paragraphs(reference_text, &reference_paragraphs);

    Embedding *input_embeddings = (Embedding*)malloc(input_paragraphs.count * sizeof(Embedding));
    Embedding *reference_embeddings = (Embedding*)malloc(reference_paragraphs.count * sizeof(Embedding));

    for (int i = 0; i < input_paragraphs.count; i++) {
        input_embeddings[i] = get_embedding(input_paragraphs.paragraphs[i], curl);
    }
    for (int j = 0; j < reference_paragraphs.count; j++) {
        reference_embeddings[j] = get_embedding(reference_paragraphs.paragraphs[j], curl);
    }

    // Allocate memory on the GPU for embeddings and results
    double *d_vec1, *d_vec2, *d_output;
    
        cudaError_t err;
err = cudaMalloc(&d_vec1, input_paragraphs.count * EMBEDDING_SIZE * sizeof(double));
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
    return -1;
}

    cudaMalloc(&d_vec1, input_paragraphs.count * EMBEDDING_SIZE * sizeof(double));
    cudaMalloc(&d_vec2, reference_paragraphs.count * EMBEDDING_SIZE * sizeof(double));
    cudaMalloc(&d_output, input_paragraphs.count * reference_paragraphs.count * sizeof(double));

    // Copy embeddings from host to GPU
    cudaMemcpy(d_vec1, input_embeddings, input_paragraphs.count * EMBEDDING_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, reference_embeddings, reference_paragraphs.count * EMBEDDING_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Free the host memory used for embeddings
    free(input_embeddings);
    free(reference_embeddings);

    // Initialize output array to zero and launch the kernel
    cudaMemset(d_output, 0, input_paragraphs.count * reference_paragraphs.count * sizeof(double));
    launch_cosine_similarity_kernel(d_vec1, d_vec2, d_output, input_paragraphs.count, reference_paragraphs.count, EMBEDDING_SIZE);

    // Allocate host memory for the results and copy from GPU
    double *h_output = (double*)malloc(input_paragraphs.count * reference_paragraphs.count * sizeof(double));
    cudaMemcpy(h_output, d_output, input_paragraphs.count * reference_paragraphs.count * sizeof(double), cudaMemcpyDeviceToHost);

    double total_similarity = 0.0;
    for (int i = 0; i < input_paragraphs.count * reference_paragraphs.count; i++) {
        total_similarity += h_output[i];
    }

    printf("Plagiarism Score: %.2f%%\n", total_similarity);

    time_t end_time = time(NULL); // Record the end time
    printf("Runtime: %ld seconds\n", end_time - start_time);

    // Cleanup
    free(h_output);
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_output);
    curl_easy_cleanup(curl);
    curl_global_cleanup();

    return 0;
}

EOF
ls -l