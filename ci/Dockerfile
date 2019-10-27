#
# The base Dockerfile that our circleci stuff runs with
#

FROM debian:stretch

# Install system libraries & tools
RUN apt-get update && \
    apt-get install -y curl wget cmake make gcc g++ pkg-config \
    libelf-dev libdw-dev binutils-dev zlib1g-dev libiberty-dev \
    libcurl4-openssl-dev libssl-dev python

ENV KCOV_VERSION 33
RUN wget https://github.com/SimonKagstrom/kcov/archive/v$KCOV_VERSION.tar.gz && \
    tar xzf v$KCOV_VERSION.tar.gz && \
    rm v$KCOV_VERSION.tar.gz && \
    cd kcov-$KCOV_VERSION && \
    mkdir build && cd build && \
    cmake .. && make && make install && \
    cd ../.. && rm -rf kcov-$KCOV_VERSION

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH "$PATH:/root/.cargo/bin"
ENV RUSTFLAGS "-C link-dead-code"

# We don't actually want to install any rust version here because they
# will change, and we want our CI to always run with the most current stable
# and beta.
#
# RUN rustup update && \
#     rustup install nightly && \
#     rustup install beta && \
#     rustup install $MSRV
#
# RUN bash -l -c 'echo $(rustc --print sysroot)/lib >> /etc/ld.so.conf'
# RUN bash -l -c 'echo /usr/local/lib >> /etc/ld.so.conf'
# RUN ldconfig
