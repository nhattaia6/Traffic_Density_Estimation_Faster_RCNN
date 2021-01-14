create database traffic_density_estimation_log;
use traffic_density_estimation_log;
create table logs(
	id int auto_increment primary key,
	date date,
    time time,
    bike int,
    car int,
    priority int,
    status varchar(20),
    image varchar(255));
    
insert into logs values
(null, '2020-12-12', '12:12:00', 4, 2, 0, 'Medium', 'link to image...');