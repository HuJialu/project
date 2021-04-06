import sabre
import a3c
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

S_INFO = 6
# bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6  #
ACTOR_LR_RATE = 0.0001
BUFFER_NORM_FACTOR = 25.0  # unit：s
TOTAL_VIDEO_CHUNKS = 199
CHUNK_TIL_VIDEO_END_CAP = 199.0
RAND_RANGE = 1000
M_IN_K = 1000.0
NN_MODEL = '../pensieve_model/pretrain_linear_reward.ckpt'
DEFAULT_QUALITY = 0


class pensieve_a3c(sabre.Abr):
    def get_quality_delay(self, segment_index):
        manifest = self.session.manifest
        bitrates = manifest.bitrates

        pensieve_file = open("pensieve_a3c_para.txt", "r")
        past_throughput_pensieve = pensieve_file.readline().replace("\n", "").split(" ")
        past_download_time_pensieve = pensieve_file.readline().replace("\n", "").split(" ")
        next_chunk_size_pensieve = pensieve_file.readline().replace("\n", "").split(" ")
        current_buffer_size_pensieve = pensieve_file.readline().replace("\n", "")
        chunks_left_pensieve = pensieve_file.readline().replace("\n", "")
        last_chunk_bitrate_pensieve = pensieve_file.readline().replace("\n", "")
        pensieve_file.close()

        for i in range(6):
            next_chunk_size_pensieve[i] = float(next_chunk_size_pensieve[i]) / M_IN_K
            past_throughput_pensieve[i] = float(past_throughput_pensieve[i]) / M_IN_K / BUFFER_NORM_FACTOR
            past_download_time_pensieve[i] = float(past_download_time_pensieve[i]) / M_IN_K / M_IN_K
        for i in range(6, 8):
            past_throughput_pensieve[i] = float(past_throughput_pensieve[i]) / M_IN_K / BUFFER_NORM_FACTOR
            past_download_time_pensieve[i] = float(past_download_time_pensieve[i]) / M_IN_K / M_IN_K

        last_chunk_bitrate_pensieve = int(float(last_chunk_bitrate_pensieve))

        # if len(bitrates) == 10:
        #     if last_chunk_bitrate_pensieve == 9:
        #         last_chunk_bitrate_pensieve = 5
        #     elif last_chunk_bitrate_pensieve == 8:
        #         last_chunk_bitrate_pensieve = 4
        #     elif last_chunk_bitrate_pensieve == 7:
        #         last_chunk_bitrate_pensieve = 3
        #     elif last_chunk_bitrate_pensieve == 6:
        #         last_chunk_bitrate_pensieve = 2
        #     elif last_chunk_bitrate_pensieve == 4:
        #         last_chunk_bitrate_pensieve = 1
        #     else:
        #         last_chunk_bitrate_pensieve = 0

        np.random.seed(42)

        with tf.Session() as sess:
            actor = a3c.ActorNetwork(sess,
                                     state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                     learning_rate=ACTOR_LR_RATE)
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            saver.restore(sess, NN_MODEL)

            state = [np.zeros((S_INFO, S_LEN))]

            state = np.roll(state, -1, axis=1)

            # # 上一个视频块比特率 old
            # state[0, 0, 7] = last_chunk_bitrate_pensieve / float(np.max(bitrates))  # last quality
            # # 当前缓冲大小
            # state[0, 1, 7] = float(current_buffer_size_pensieve) / BUFFER_NORM_FACTOR  # 10 sec
            # # 未来视频块不同质量大小
            # state[0, 2, :] = np.array(past_download_time_pensieve)  # kilo byte / ms
            # # 过去8个视频块对应网络吞吐量
            # state[0, 3, :] = past_throughput_pensieve  # 10 sec
            # # 过去8个视频块下载时间
            # state[0, 4, :A_DIM] = next_chunk_size_pensieve  # mega byte
            # # 剩余视频块数量
            # state[0, 5, 7] = float(chunks_left_pensieve) / float(CHUNK_TIL_VIDEO_END_CAP)

            # 上一个视频块比特率 new
            state[0, 0, 7] = bitrates[last_chunk_bitrate_pensieve] / float(np.max(bitrates))  # last quality
            # 当前缓冲大小
            state[0, 1, 7] = float(current_buffer_size_pensieve) / BUFFER_NORM_FACTOR  # 10 sec
            # 未来视频块不同质量大小
            state[0, 2, :] = np.array(past_download_time_pensieve)  # kilo byte / ms
            # 过去8个视频块对应网络吞吐量
            state[0, 3, :] = past_throughput_pensieve  # 10 sec
            # 过去8个视频块下载时间
            state[0, 4, :A_DIM] = next_chunk_size_pensieve  # mega byte
            # 剩余视频块数量
            state[0, 5, 7] = float(chunks_left_pensieve) / float(CHUNK_TIL_VIDEO_END_CAP)

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

            if len(bitrates) == 10:
                if bit_rate == 5:
                    quality = 9
                elif bit_rate == 4:
                    quality = 8
                elif bit_rate == 3:
                    quality = 7
                elif bit_rate == 2:
                    quality = 6
                elif bit_rate == 1:
                    quality = 4
                else:
                    quality = 0
            else:
                quality = bit_rate

        return (quality, 0)
