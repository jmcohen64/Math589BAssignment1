/*
#include <cmath>
#include <vector>
#include "energy.hpp"

void compute_gradient(double* positions, double* gradients, int n_beads, double epsilon, double sigma, double b, double k_b) {
    // Initialize gradients to zero
    for (int i = 0; i < n_beads * 3; ++i) {
        gradients[i] = 0.0;
    }

    // Compute bond potential gradient
    for (int i = 0; i < n_beads - 1; ++i) {
        int idx1 = i * 3;
        int idx2 = (i + 1) * 3;
        
        double dx = positions[idx2] - positions[idx1];
        double dy = positions[idx2 + 1] - positions[idx1 + 1];
        double dz = positions[idx2 + 2] - positions[idx1 + 2];
        double r = sqrt(dx * dx + dy * dy + dz * dz);

        if (r > 1e-12) {
            double force_mag = 2 * k_b * (r - b) / r;

            // Apply forces to both beads
            gradients[idx1] -= force_mag * dx;
            gradients[idx1 + 1] -= force_mag * dy;
            gradients[idx1 + 2] -= force_mag * dz;

            gradients[idx2] += force_mag * dx;
            gradients[idx2 + 1] += force_mag * dy;
            gradients[idx2 + 2] += force_mag * dz;
        }
    }

    // Compute Lennard-Jones potential gradient
    for (int i = 0; i < n_beads; ++i) {
        for (int j = i + 1; j < n_beads; ++j) {
            int idx1 = i * 3;
            int idx2 = j * 3;

            double dx = positions[idx2] - positions[idx1];
            double dy = positions[idx2 + 1] - positions[idx1 + 1];
            double dz = positions[idx2 + 2] - positions[idx1 + 2];
            double r = sqrt(dx * dx + dy * dy + dz * dz);

            if (r > 1e-12) {
                double sr6 = pow(sigma / r, 6);
                double force_mag = (24 * epsilon / (r * r)) * sr6 * (2 * sr6 - 1);  // Derivative of Lennard-Jones potential

                // Apply forces to both beads
                gradients[idx1] -= force_mag * dx;
                gradients[idx1 + 1] -= force_mag * dy;
                gradients[idx1 + 2] -= force_mag * dz;

                gradients[idx2] += force_mag * dx;
                gradients[idx2 + 1] += force_mag * dy;
                gradients[idx2 + 2] += force_mag * dz;
            }
        }
    }
}
*/
#include <cmath>
#include <vector>
#include "energy.hpp"

void compute_gradient(double* positions, double* gradients, int n_beads, double epsilon, double sigma, double b, double k_b) {
    // Initialize gradients to zero
    for (int i = 0; i < n_beads * 3; ++i) {
        gradients[i] = 0.0;
    }

    // Compute bond force contributions
    for (int i = 0; i < n_beads - 1; ++i) {
        int idx1 = i * 3;
        int idx2 = (i + 1) * 3;

        double dx = positions[idx2] - positions[idx1];
        double dy = positions[idx2 + 1] - positions[idx1 + 1];
        double dz = positions[idx2 + 2] - positions[idx1 + 2];

        double r = sqrt(dx * dx + dy * dy + dz * dz);
        double force_magnitude = -2 * k_b * (r - b) / r;  // Negative gradient of bond potential

        gradients[idx1] += force_magnitude * dx;
        gradients[idx1 + 1] += force_magnitude * dy;
        gradients[idx1 + 2] += force_magnitude * dz;

        gradients[idx2] -= force_magnitude * dx;
        gradients[idx2 + 1] -= force_magnitude * dy;
        gradients[idx2 + 2] -= force_magnitude * dz;
    }

    // Compute Lennard-Jones force contributions
    for (int i = 0; i < n_beads; ++i) {
        for (int j = i + 1; j < n_beads; ++j) {
            int idx1 = i * 3;
            int idx2 = j * 3;

            double dx = positions[idx2] - positions[idx1];
            double dy = positions[idx2 + 1] - positions[idx1 + 1];
            double dz = positions[idx2 + 2] - positions[idx1 + 2];

            double r = sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1e-12) continue;  // Avoid division by zero

            double sr6 = pow(sigma / r, 6);
            double force_magnitude = 24 * epsilon * (2 * sr6 * sr6 - sr6) / (r * r);  // Negative gradient of LJ potential

            gradients[idx1] += force_magnitude * dx;
            gradients[idx1 + 1] += force_magnitude * dy;
            gradients[idx1 + 2] += force_magnitude * dz;

            gradients[idx2] -= force_magnitude * dx;
            gradients[idx2 + 1] -= force_magnitude * dy;
            gradients[idx2 + 2] -= force_magnitude * dz;
        }
    }
}

