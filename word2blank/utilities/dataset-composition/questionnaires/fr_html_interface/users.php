<?php

include_once('db.php');
header('Content-Type: text/html; charset=ISO-8859-1');
$users = array();
$conecta = get_db_connection();
$sql = "SELECT anotador, name, surname FROM " . $prefix . "users;";
$result = mysql_query($sql, $conecta);
if($result){
    while($consulta = mysql_fetch_array($result)) {            
        $users[$consulta[0]] = $consulta[1] . " " . $consulta[2];            
    }
}
mysql_close($conecta);

?>
