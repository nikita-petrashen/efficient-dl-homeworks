parts = ["a", "b", "c", "d"]


def ring_allreduce_test():
    size = 2
    sends = [[f"{part}{rank}" for part in parts[:size]] for rank in range(size)]
    print("######### DISTRIBUTING PARTS STAGE #########")
    for i in range(size-1):
        print(f"step {i}")
        for rank in range(size):
            send_rank = (rank + 1) % size
            own = sends[rank]
            nxt = sends[send_rank]
            send_i = (rank - i) % size
            nxt_recv_i = (rank - i) % size
            nxt[nxt_recv_i] += own[send_i]
            print(f"\trank {rank} sent {own[send_i]} to {send_rank}")

        for i in range(size):
            print(f"rank {i} has data {sends[i]}")

        print("\n-------------\n")

    print("######### DISTRIBUTING AVG STAGE #########")

    for i in range(size-1):
        print(f"step {i}")
        for rank in range(size):
            send_rank = (rank + 1) % size
            own = sends[rank]
            nxt = sends[send_rank]
            send_i = (rank - i + 1) % size
            nxt_recv_i = (rank - i + 1) % size
            nxt[nxt_recv_i] = own[send_i]
            print(f"\trank {rank} sent {own[send_i]} to {send_rank}")

        for i in range(size):
            print(f"rank {i} has data {sends[i]}")

        print("\n-------------\n")


if __name__ == "__main__":
    ring_allreduce_test()