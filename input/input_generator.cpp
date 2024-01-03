#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

int main(int argc, char* argv[]) {
    // Check if the number of arguments is correct
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <N>" << std::endl;
        return 1;
    }

    // Get the value of N from command line arguments
    int N = std::atoi(argv[1]);

    // Open the output file
    std::ofstream outFile("input_" + std::to_string(N) + ".txt");
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing." << std::endl;
        return 1;
    }

    // Seed the random number generator with the current time
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Write N to the file
    outFile << N << std::endl;

    // Generate and write N 3-dimensional integers in the range -1000 to 1000 to the file
    for (int i = 0; i < N; ++i) {
        int x = std::rand() % 2001 - 1000;  // Range: -1000 to 1000
        int y = std::rand() % 2001 - 1000;
        int z = std::rand() % 2001 - 1000;
        outFile << x << " " << y << " " << z << std::endl;
    }

    // Close the file
    outFile.close();

    std::cout << "File input_" << N << ".txt generated successfully." << std::endl;

    return 0;
}
