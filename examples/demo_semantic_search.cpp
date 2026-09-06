#include "vectordb/engine/hnsw_index.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

struct Movie {
    uint64_t id;
    std::string title;
    std::string genre;
    std::string description;
    vectordb::Vector embedding; // 8-D semantic features: [Space, Crime, Romance, Action, Nature, Tech, Family, Mystery]
};

int main() {
    std::cout << "========================================================================\n";
    std::cout << "  High-Throughput Vector Search Engine — Semantic Search Demo (C++20)\n";
    std::cout << "========================================================================\n\n";

    vectordb::EngineConfig config;
    config.dimension = 8;
    config.metric = vectordb::MetricType::COSINE;
    config.hnsw_m = 16;
    config.hnsw_ef_construction = 100;
    config.hnsw_ef_search = 50;

    vectordb::HNSWIndex index(config);

    // 8 Semantic dimensions:
    // [0] Space, [1] Crime, [2] Romance, [3] Action, [4] Nature, [5] Tech, [6] Family, [7] Mystery
    std::vector<Movie> catalog = {
        {1, "Interstellar", "Sci-Fi", "Astronauts travel through a wormhole in space to save humanity.",
         {0.95f, 0.05f, 0.20f, 0.40f, 0.10f, 0.90f, 0.60f, 0.30f}},

        {2, "The Dark Knight", "Action/Crime", "Batman battles the Joker to save Gotham City from chaos.",
         {0.05f, 0.95f, 0.10f, 0.90f, 0.00f, 0.50f, 0.10f, 0.85f}},

        {3, "The Godfather", "Crime/Drama", "An organized crime dynasty transfers control to its reluctant son.",
         {0.00f, 0.98f, 0.20f, 0.40f, 0.05f, 0.00f, 0.70f, 0.60f}},

        {4, "The Martian", "Sci-Fi", "An astronaut becomes stranded on Mars and must survive using science.",
         {0.92f, 0.00f, 0.05f, 0.35f, 0.20f, 0.95f, 0.10f, 0.20f}},

        {5, "Titanic", "Romance/Drama", "Two lovers from different social classes meet aboard an ill-fated ship.",
         {0.00f, 0.05f, 0.98f, 0.30f, 0.40f, 0.10f, 0.30f, 0.10f}},

        {6, "Planet Earth", "Documentary", "An incredible journey exploring nature, wildlife, and ecosystems.",
         {0.05f, 0.00f, 0.10f, 0.10f, 0.98f, 0.05f, 0.50f, 0.10f}},

        {7, "Blade Runner 2049", "Sci-Fi/Mystery", "A blade runner unearths a secret that could plunge society into chaos.",
         {0.60f, 0.70f, 0.25f, 0.65f, 0.05f, 0.95f, 0.10f, 0.92f}},

        {8, "Finding Nemo", "Animation/Family", "A clownfish searches the ocean across reefs to find his lost son.",
         {0.00f, 0.05f, 0.20f, 0.30f, 0.85f, 0.00f, 0.98f, 0.20f}},

        {9, "Inception", "Sci-Fi/Action", "Thieves enter the human subconscious to steal corporate secrets.",
         {0.30f, 0.75f, 0.30f, 0.80f, 0.00f, 0.88f, 0.40f, 0.95f}},

        {10, "Pride and Prejudice", "Romance", "Sparks fly when spirited Elizabeth Bennet meets single Mr. Darcy.",
         {0.00f, 0.00f, 0.95f, 0.05f, 0.30f, 0.00f, 0.60f, 0.20f}}
    };

    std::cout << "[1/3] Indexing " << catalog.size() << " movies into the HNSW graph..." << std::endl;
    for (const auto& m : catalog) {
        index.insert(m.id, m.embedding);
    }
    std::cout << "      Done! Graph built with AVX2 distance acceleration.\n\n";

    // Demo query scenarios
    struct QueryScenario {
        std::string query_text;
        vectordb::Vector query_vector;
    };

    std::vector<QueryScenario> scenarios = {
        {"'astronaut stranded in deep space with technology'", {0.92f, 0.00f, 0.05f, 0.30f, 0.10f, 0.92f, 0.20f, 0.25f}},
        {"'mafia crime boss underworld detective mystery'",    {0.00f, 0.95f, 0.10f, 0.60f, 0.00f, 0.10f, 0.20f, 0.85f}},
        {"'passionate romantic lovers drama aboard ship'",     {0.00f, 0.05f, 0.96f, 0.20f, 0.30f, 0.05f, 0.40f, 0.15f}},
        {"'wild animals ocean reef family journey'",           {0.00f, 0.00f, 0.15f, 0.20f, 0.92f, 0.00f, 0.90f, 0.10f}}
    };

    std::cout << "[2/3] Running Semantic Similarity Searches:\n";
    std::cout << "------------------------------------------------------------------------\n";

    for (const auto& scenario : scenarios) {
        std::cout << "\n>>> USER QUERY: " << scenario.query_text << "\n";

        auto start = std::chrono::high_resolution_clock::now();
        auto results = index.search(scenario.query_vector, 3);
        auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - start).count();

        std::cout << "    Search Latency: " << elapsed_ns / 1000.0 << " microseconds\n";
        std::cout << "    Top Matches:\n";

        for (size_t rank = 0; rank < results.size(); ++rank) {
            const auto& r = results[rank];
            const Movie* matched = nullptr;
            for (const auto& m : catalog) {
                if (m.id == r.id) { matched = &m; break; }
            }
            if (!matched) continue;

            float similarity_pct = std::max(0.0f, (1.0f - r.distance) * 100.0f);
            std::cout << "      " << (rank + 1) << ". " << std::left << std::setw(22) << matched->title
                      << " [" << matched->genre << "] "
                      << "Match: " << std::fixed << std::setprecision(1) << similarity_pct << "%\n"
                      << "         \"" << matched->description << "\"\n";
        }
    }

    std::cout << "\n[3/3] Semantic Search Demo COMPLETED successfully!\n";
    std::cout << "========================================================================\n";
    return 0;
}
