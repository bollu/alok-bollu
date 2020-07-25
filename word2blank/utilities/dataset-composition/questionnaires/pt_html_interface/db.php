<?php

function get_db_connection() {
    //TODO Replace host, database name, username and password for your DB
    $conecta = mysql_connect("db-host", "db-username", "db-password") or print (mysql_error()); 
    mysql_select_db("db-name", $conecta) or print(mysql_error());
    return $conecta;
}
?>

