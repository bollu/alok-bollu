<?php
$prefix = "mturk_pt_";
if(isset($_POST['ok'])){
    include_once('users.php');
    $anno = strtolower(mysql_escape_string($_POST['annotator']));
    echo $users[0];
    if(isset($users[$anno])){
        setcookie("annotator", $anno);
        $actual_link = "http://$_SERVER[HTTP_HOST]$_SERVER[REQUEST_URI]";
        $actual_link = str_replace("http://wprojs-php/~","http://www.inf.ufrgs.br/",$actual_link);
	      $actual_link = str_replace("index.php","",$actual_link);
        echo "<META http-equiv='refresh' content='0;URL=" . $actual_link . "pagina.php'> ";
        echo "Se você não for redirecionado automaticamente <a href='" . $actual_link ."'>clique aqui</a>";   
    }else{        
        require('login.php');
    }
}else{
    require('login.php');
}

?>
