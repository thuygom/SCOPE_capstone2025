use followSupport;

CREATE TABLE VideoStat (
    video_id VARCHAR(50),
    upload_date DATETIME,
    date DATE,
    view_count INT,
    like_count INT,
    comment_count INT,
    subscriber_count INT,
    channel_id VARCHAR(50),
    channel_title VARCHAR(100),
    channel_description TEXT null,
    topic_categories TEXT null,
    title VARCHAR(200) null,
    description TEXT null,
    tags TEXT null,
    thumbnails VARCHAR(255),
    PRIMARY KEY (video_id, date)
);

CREATE TABLE Comments (
    comment_id INT AUTO_INCREMENT PRIMARY KEY,
    comment TEXT,
    author VARCHAR(100),
    date DATETIME,
    num_likes INT,
    video_id VARCHAR(50),
    FOREIGN KEY (video_id) REFERENCES VideoStat(video_id)
);

select * from VideoStat;
select * from Comments;