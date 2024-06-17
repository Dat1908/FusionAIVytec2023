$(document).ready(function () {
  $("#upload_button").click(function () {
    $("#fileinput").trigger("click");
  });

  $("#fileinput").change(function () {
    $("#form").submit();
    $("#upload_hint").text(
      "Đang tải file " + $("#fileinput")[0].files[0].name + " lên server..."
    );
  });
});
