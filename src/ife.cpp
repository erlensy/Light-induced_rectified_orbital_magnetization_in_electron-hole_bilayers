#include <armadillo>
#include <iostream>
#include <tuple>
#include <string>
#include <chrono>
#include "math.h"
#include "struve.h"

auto program_start = std::chrono::steady_clock::now();

struct Config {
    double m_e, m_h, r_0_e, r_0_h, epsilon_p, epsilon_m, d;
    double E_0, omega, t_ramp, alpha;
    double x_max, y_max, t_max;
    int x_n, y_n, t_n, t_save_n;
};

struct Params {
    double hbar = 1.05457182 * 1e-34;
    double e = 1.60217663 * 1e-19;
    double epsilon_0 = 8.8541878188 * 1e-12;
    double pi = 3.1415926535;
    double m_0 = 9.1093837139 * 1e-31;

    double mu, r_0, kappa, m_e, m_h;
    double S_r, S_t, S_E, g;
    double d, E_0, omega, t_ramp, x_max, y_max, dx, dy, dt, alpha;
    int x_n, y_n, t_n, t_save_n;
};

void update_params_from_config(Params& params, Config& config) {
    params.mu = config.m_e * config.m_h / (config.m_e + config.m_h);
    params.m_e = config.m_e;
    params.m_h = config.m_h;
    params.r_0 = config.r_0_e + config.r_0_h;
    params.kappa = (config.epsilon_p + config.epsilon_m) / 2.0;

    params.S_r = params.r_0 / params.kappa;
    params.S_t = 2 * params.mu * params.S_r * params.S_r / params.hbar;
    params.S_E = params.hbar * params.hbar / (2 * params.mu * params.e * params.S_r * params.S_r * params.S_r);
    params.g = params.mu * params.e * params.e * params.S_r * params.S_r / (4 * params.hbar * params.hbar * params.epsilon_0 * params.r_0);

    params.d = config.d / params.S_r;
    params.E_0 = config.E_0 / params.S_E;
    params.omega = config.omega * params.S_t;
    params.t_ramp = config.t_ramp / params.S_t;
    params.alpha = config.alpha;
    params.x_max = config.x_max / params.S_r;
    params.y_max = config.y_max / params.S_r;
    params.x_n = config.x_n;
    params.y_n = config.y_n;
    params.t_n = config.t_n;
    params.t_save_n = config.t_save_n;
    params.dx = params.x_max / (params.x_n - 1);
    params.dy = params.y_max / (params.y_n - 1);
    params.dt = (config.t_max / (params.t_n - 1)) / params.S_t;
}

arma::sp_mat finite_diff_laplacian_1d(int n, double d) {
    arma::sp_mat L(n, n);
    L.diag(0).fill(-2.0);
    L.diag(1).fill(1.0);
    L.diag(-1).fill(1.0);
    L /= (d * d);
    return L;
}

void finite_diff_first_x(arma::cx_mat& out, arma::cx_mat& in, int c1, int c2x, int c2y, double dx) {
    using arma::span;
    out(span(c1 + 1, c2x - 1), span(c1, c2y)) = (in(span(c1 + 2, c2x), span(c1, c2y)) - in(span(c1, c2x - 2), span(c1, c2y))) / (2.0 * dx);
    out(c1, span(c1, c2y)) = (in(c1 + 1, span(c1, c2y)) - in(c1, span(c1, c2y))) / dx;
    out(c2x, span(c1, c2y)) = (in(c2x, span(c1, c2y)) - in(c2x - 1, span(c1, c2y))) / dx;
}

void finite_diff_first_y(arma::cx_mat& out, arma::cx_mat& in, int c1, int c2x, int c2y, double dy) {
    using arma::span;
    out(span(c1, c2x), span(c1 + 1, c2y - 1)) = (in(span(c1, c2x), span(c1 + 2, c2y)) - in(span(c1, c2x), span(c1, c2y - 2))) / (2.0 * dy);
    out(span(c1, c2x), c1) = (in(span(c1, c2x), c1 + 1) - in(span(c1, c2x), c1)) / dy;
    out(span(c1, c2x), c2y) = (in(span(c1, c2x), c2y) - in(span(c1, c2x), c2y - 1)) / dy;
}

void apply_second_deriv_x(arma::mat& out, arma::mat& in, int c1, int c2x, int c2y, double dx) {
    using arma::span;
    out(span(c1, c2x), span(c1, c2y)) -= (in(span(c1 + 1, c2x + 1), span(c1, c2y)) - 2.0 * in(span(c1, c2x), span(c1, c2y)) + in(span(c1 - 1, c2x - 1), span(c1, c2y))) / (dx * dx);
}

void apply_second_deriv_y(arma::mat& out, arma::mat& in, int c1, int c2x, int c2y, double dy) {
    using arma::span;
    out(span(c1, c2x), span(c1, c2y)) -= (in(span(c1, c2x), span(c1 + 1, c2y + 1)) -2.0 * in(span(c1, c2x), span(c1, c2y)) + in(span(c1, c2x), span(c1 - 1, c2y - 1))) / (dy * dy);
}

void normalize_psi(arma::cx_mat& psi, Params& params) {
    double norm = std::sqrt(arma::accu(arma::square(arma::abs(psi))) * params.dx * params.dy);
    psi /= norm;
}

double rytova_keldysh_potential(double x, double y, Params& params) {
    double arg = sqrt(x * x + y * y + params.d * params.d);
    return -params.g * (struveh0(arg) - yn(0, arg));
}

std::tuple<arma::mat, arma::mat, arma::mat> make_grids_and_potential(Params& params) {
    arma::vec x_list = arma::linspace(-params.x_max / 2.0, params.x_max / 2.0, params.x_n);
    arma::vec y_list = arma::linspace(-params.y_max / 2.0, params.y_max / 2.0, params.y_n);

    arma::mat x_mat = arma::repmat(x_list, 1, params.y_n);
    arma::mat y_mat = arma::repmat(y_list.t(), params.x_n, 1);

    arma::mat v_mat(params.x_n, params.y_n, arma::fill::zeros);
    for (int i = 0; i < params.x_n; i++) {
        for (int j = 0; j < params.y_n; j++) {
            v_mat(i, j) = rytova_keldysh_potential(x_mat(i, j), y_mat(i, j), params);
        }
    }

    return {v_mat, x_mat, y_mat};
}

std::tuple<double, double> electric_field(double t, Params& params) {
    double ramp = (1 + erf((t - params.t_ramp / 2) / (params.t_ramp / 6))) / 2;
    double E_x = params.E_0 * cos(params.omega * t) * ramp;
    double E_y = params.E_0 * sin(params.omega * t + params.alpha) * ramp;
    return {E_x, E_y};
}

void ham(arma::mat& ham_psi, arma::mat& psi, double t, arma::mat& v_mat, arma::mat& x_mat, arma::mat& y_mat, Params& params) {
    auto [E_x, E_y] = electric_field(t, params);
    ham_psi = (v_mat + E_x * x_mat + E_y * y_mat) % psi;
    apply_second_deriv_x(ham_psi, psi, 1, params.x_n - 2, params.y_n - 2, params.dx);
    apply_second_deriv_y(ham_psi, psi, 1, params.x_n - 2, params.y_n - 2, params.dy);
}

arma::cx_double compute_lz(arma::cx_mat& psi1, arma::cx_mat& psi2, arma::mat& x_mat, arma::mat& y_mat, Params& params) {
    arma::cx_mat dx_psi2(psi2.n_rows, psi2.n_cols, arma::fill::zeros);
    finite_diff_first_x(dx_psi2, psi2, 1, params.x_n - 2, params.y_n - 2, params.dx);

    arma::cx_mat dy_psi2(psi2.n_rows, psi2.n_cols, arma::fill::zeros);
    finite_diff_first_y(dy_psi2, psi2, 1, params.x_n - 2, params.y_n - 2, params.dy);

    arma::cx_mat lz_psi2 = -1.0 * arma::cx_double(0.0, 1.0) * (x_mat % dy_psi2 - y_mat % dx_psi2);
    arma::cx_double lz = arma::accu(arma::conj(psi1) % lz_psi2) * params.dx * params.dy;
    return lz;
}

arma::sp_mat create_H_0(arma::mat& v_mat, arma::mat& x_mat, arma::mat& y_mat, Params& params) {
    int c1 = 1;
    int c2x = params.x_n - 2;
    int c2y = params.y_n - 2;
    arma::mat v_int = v_mat(arma::span(c1, c2x), arma::span(c1, c2y));

    arma::sp_mat L_xx = finite_diff_laplacian_1d(c2x, params.dx);
    arma::sp_mat L_yy = finite_diff_laplacian_1d(c2y, params.dy);

    arma::sp_mat id_x = arma::speye<arma::sp_mat>(c2x, c2x);
    arma::sp_mat id_y = arma::speye<arma::sp_mat>(c2y, c2y);

    arma::sp_mat T = -arma::kron(id_y, L_xx) - arma::kron(L_yy, id_x);
    arma::vec v = arma::vectorise(v_int);
    arma::sp_mat V(c2x * c2y, c2x * c2y);
    V.diag() = v;
    arma::sp_mat H_0 = T + V;

    return H_0;
}

arma::cx_mat init_state(int n, arma::mat& v_mat, arma::mat& x_mat, arma::mat& y_mat, Params& params) {
    arma::sp_mat H_0 = create_H_0(v_mat, x_mat, y_mat, params);
    arma::vec eigval;
    arma::mat eigvec;
    arma::eigs_sym(eigval, eigvec, H_0, n + 1, "sa");

    arma::mat v_n = arma::reshape(eigvec.col(n), params.x_n - 2, params.y_n - 2);
    arma::cx_mat vv_n = arma::conv_to<arma::cx_mat>::from(v_n);
    arma::cx_mat psi = arma::zeros<arma::cx_mat>(params.x_n, params.y_n);
    psi(arma::span(1, params.x_n - 2), arma::span(1, params.y_n - 2)) = vv_n;
    normalize_psi(psi, params);

    return psi;
}

arma::cx_vec time_evolve(arma::cx_mat& psi_0, arma::mat& v_mat, arma::mat& x_mat, arma::mat& y_mat, Params& params) {
    arma::cx_mat psi = psi_0;

    arma::mat psi_u = arma::real(psi);
    arma::mat psi_v = arma::imag(psi);
    arma::mat ham_psi = arma::zeros<arma::mat>(params.x_n, params.y_n);
    ham(ham_psi, psi_v, 0, v_mat, x_mat, y_mat, params);
    psi_u += params.dt / 2 * ham_psi;

    arma::cx_vec orbital_mag = arma::zeros<arma::cx_vec>(params.t_n);
    for (int i = 1; i < params.t_n - 1; i++) {
        double t = params.dt * i;
        ham(ham_psi, psi_u, t + 0.5 * params.dt, v_mat, x_mat, y_mat, params);
        psi_v -= params.dt * ham_psi;
        ham(ham_psi, psi_v, t + params.dt, v_mat, x_mat, y_mat, params);
        psi_u += params.dt * ham_psi;
        psi = psi_u + arma::cx_double(0.0, 1.0) * psi_v;
        orbital_mag(i) = compute_lz(psi, psi, x_mat, y_mat, params);
    }
    return orbital_mag * params.m_0 * (1.0 / params.m_h - 1.0 / params.m_e);
}

void save_wavefunctions() {
    int n = 6;

    double minutes = std::chrono::duration<double, std::ratio<60>>(std::chrono::steady_clock::now() - program_start).count();
    std::cout << "\rt=" << minutes << ":\tstarted wavefunctions\n";

    Config config;
    Params params;
    config.m_e = 0.8 * params.m_0;
    config.m_h = 0.4 * params.m_0;
    config.r_0_e = config.r_0_h = 4.0e-9;
    config.x_max = config.y_max = 500e-9;
    config.x_n = config.y_n = 1000;
    config.epsilon_p = config.epsilon_m = 2.5;
    config.d = 1e-9;
    update_params_from_config(params, config);

    auto [v_mat, x_mat, y_mat] = make_grids_and_potential(params);
    arma::sp_mat H_0 = create_H_0(v_mat, x_mat, y_mat, params);
    arma::vec eigval;
    arma::mat eigvec;
    arma::eigs_sym(eigval, eigvec, H_0, n, "sa");
    double dim_factor = params.hbar * params.hbar / (2 * params.mu * params.S_r * params.S_r);
    eigval *= dim_factor / params.e * 1e3;
    eigval.save("../data/wavefunctions_energies.bin", arma::raw_binary);

    for (int state = 0; state < n; state++) {
        arma::mat state_inner = arma::reshape(eigvec.col(state), params.x_n - 2, params.y_n - 2);
        arma::cx_mat psi = arma::zeros<arma::cx_mat>(params.x_n, params.y_n);
        psi(arma::span(1, params.x_n - 2), arma::span(1, params.y_n - 2)) = arma::conv_to<arma::cx_mat>::from(state_inner);
        normalize_psi(psi, params);
        std::string filename = "../data/psi_" + std::to_string(state) + ".bin";
        psi.save(filename, arma::raw_binary);

        minutes = std::chrono::duration<double, std::ratio<60>>(std::chrono::steady_clock::now() - program_start).count();
        std::cout << "\rt=" << minutes << ":\tfinished " << state + 1 << "/" << n << " wavefunctions" << std::flush;
    }
}

void binding_energy() {
    double minutes = std::chrono::duration<double, std::ratio<60>>(std::chrono::steady_clock::now() - program_start).count();
    std::cout << "\n\nt=" << minutes << ":\t started binding energies\n";

    Config config;
    Params params;
    config.m_e = 0.8 * params.m_0;
    config.m_h = 0.4 * params.m_0;
    config.r_0_e = config.r_0_h = 4.0e-9;
    config.x_max = config.y_max = 500e-9;
    config.x_n = config.y_n = 1000;

    double d_min = 0.5e-9;
    double d_max = 10.0e-9;
    int d_n = 50;
    arma::vec d_list = arma::linspace(d_min, d_max, d_n);

    arma::mat output = arma::zeros<arma::mat>(d_n, 2);
    for (int k = 1; k < 5; k++) {
        config.epsilon_p = config.epsilon_m = k;
        for (int i = 0; i < d_n; i++) {
            config.d = d_list(i);
            update_params_from_config(params, config);

            auto [v_mat, x_mat, y_mat] = make_grids_and_potential(params);
            arma::sp_mat H_0 = create_H_0(v_mat, x_mat, y_mat, params);
            arma::vec eigval;
            arma::eigs_sym(eigval, H_0, 1, "sa");
            double dim_factor = params.hbar * params.hbar / (2 * params.mu * params.S_r * params.S_r);
            double energy = eigval(0) * dim_factor / params.e * 1e3;
            output(i, 0) = d_list(i);
            output(i, 1) = energy;

            minutes = std::chrono::duration<double, std::ratio<60>>(std::chrono::steady_clock::now() - program_start).count();
            std::cout << "\rt=" << minutes << ":\t finished binding energy for kappa_n=" << k << "/4" << " and d_n=" << i + 1 << "/" << d_n << std::flush;
        }
        std::string filename = "../data/binding_energy_kappa=" + std::to_string(k) + ".bin";
        output.save(filename, arma::raw_binary);
    }
}


void mag_of_t() {
    double minutes = std::chrono::duration<double, std::ratio<60>>(std::chrono::steady_clock::now() - program_start).count();
    std::cout << "\n\nt=" << minutes << ":\t started magnetization as function of t\n";

    Config config;
    Params params;
    config.m_e = 0.8 * params.m_0;
    config.m_h = 0.4 * params.m_0;
    config.r_0_e = config.r_0_h = 4.0e-9;
    config.x_max = config.y_max = 250e-9;
    config.x_n = config.y_n = 100;
    config.epsilon_p = config.epsilon_m = 4.0;
    config.d = 5e-9;
    config.E_0 = 1.0e5;
    config.omega = 1.0e12;
    config.t_n = 100000;
    config.t_ramp = 3 * 2 * params.pi / config.omega;
    config.t_max = 10 * 2 * params.pi / config.omega;
    update_params_from_config(params, config);
    auto [v_mat, x_mat, y_mat] = make_grids_and_potential(params);
    std::vector<std::string> labels = {"LHCP", "LP", "RHCP"};
    for (int i = 0; i < 3; i++) {
        config.alpha = i * params.pi / 2;
        update_params_from_config(params, config);
        arma::cx_mat psi = init_state(0, v_mat, x_mat, y_mat, params);
        arma::cx_vec orbital_mag = time_evolve(psi, v_mat, x_mat, y_mat, params);
        std::string filename = "../data/mag_of_t_" + labels[i] + ".bin";
        orbital_mag.save(filename, arma::raw_binary);

        minutes = std::chrono::duration<double, std::ratio<60>>(std::chrono::steady_clock::now() - program_start).count();
        std::cout << "\rt=" << minutes << ":\t finished magnetization as function of t for polarization_n=" << i + 1 << "/3" << std::flush;
    }
}

void mag_of_d() {
    double minutes = std::chrono::duration<double, std::ratio<60>>(std::chrono::steady_clock::now() - program_start).count();
    std::cout << "\n\nt=" << minutes << ":\t started magnetization as function of d\n";

    Config config;
    Params params;
    config.m_e = 0.8 * params.m_0;
    config.m_h = 0.4 * params.m_0;
    config.r_0_e = config.r_0_h = 4.0e-9;
    config.x_max = config.y_max = 250e-9;
    config.x_n = config.y_n = 100;
    config.E_0 = 1.0e5;
    config.alpha = 0;
    config.omega = 1.0e12;
    config.t_n = 100000;
    config.t_ramp = 3 * 2 * params.pi / config.omega;
    config.t_max = 10 * 2 * params.pi / config.omega;

    int ds_n = 50;
    arma::vec ds = arma::linspace(0.5e-9, 10e-9, ds_n);
    for (int k = 1; k < 5; k++) {
        config.epsilon_p = k;
        config.epsilon_m = k;
        arma::mat avg_orbital_mag = arma::zeros<arma::mat>(ds_n, 2);
        for (int i = 0; i < ds_n; i++) {
            config.d = ds(i);
            update_params_from_config(params, config);
            auto [v_mat, x_mat, y_mat] = make_grids_and_potential(params);
            arma::cx_mat psi = init_state(0, v_mat, x_mat, y_mat, params);
            arma::cx_vec orbital_mag = time_evolve(psi, v_mat, x_mat, y_mat, params);
            avg_orbital_mag(i, 0) = ds(i);
            avg_orbital_mag(i, 1) = arma::mean(arma::real(orbital_mag.subvec(int(params.t_n / 3), orbital_mag.n_elem - 1))); 

            minutes = std::chrono::duration<double, std::ratio<60>>(std::chrono::steady_clock::now() - program_start).count();
            std::cout << "\rt=" << minutes << ":\t finished magnetization as function of d for kappa_n=" << k << "/4" << " and d_n=" << i + 1 << "/" << ds_n << std::flush;
        }
        std::string filename = "../data/mag_of_d_kappa=" + std::to_string(k) + ".bin";
        avg_orbital_mag.save(filename, arma::raw_binary);
    }
}

int main() {
    // fig 4 (a) in paper
    binding_energy();

    // fig 4 (b) in paper
    save_wavefunctions();

    // fig 5 (a) in paper
    mag_of_t();

    // fig 5 (b) in paper
    mag_of_d();
}
